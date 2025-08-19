import cv2
import numpy as np
import torch
import os
from scipy.ndimage import uniform_filter1d
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

# ======= CONFIG ============
YOLO_OBJECT_MODEL_PATH = r"runs\detect\sam\8m50e\weights\best.pt"  # โมเดลตรวจจับวัตถุ เช่น Airbag
YOLO_NAME_MODEL_PATH = r"runs\detect\Circle_name\8m50e\weights\best.pt"  # โมเดลอ่านชื่อ เช่น FR1, FR2, RE3
SAM_CHECKPOINT = r"Model\sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"
VIDEO_PATH = r"DATA\Room\N1WB-E042D94-AEDYACB25149110311_23_Side.avi"
VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PATH = fr"Output\{VIDEO_NAME}_Timing_Detection.mp4"
SCREENSHOT_DIR = os.path.join("Output", "Screenshots", VIDEO_NAME)
CONFIDENCE_THRESHOLD = 0.5
MOTION_THRESHOLD = 1500
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

SMOOTH_SIZE = 3
PLATEAU_ALPHA = 0.98

# ======= ROBUST CIRCLE CONFIG =========
MAX_FRAMES_FOR_CIRCLE_SEARCH = 80   # ลองหาให้เต็มที่กี่เฟรมก่อนใช้ fallback ชื่อ
CIRCLE_MIN_DIST = 45                # เดิม 60
CIRCLE_PARAM_SETS = [
    # ชุดมาตรฐาน (เดิมใกล้เคียง)
    dict(dp=1.2, param1=80, param2=30, minRadius=8,  maxRadius=60),
    # ไวต่อขอบมากขึ้น/รัศมีใหญ่ขึ้น
    dict(dp=1.0, param1=70, param2=24, minRadius=8,  maxRadius=72),
    # ไวขึ้นอีก ลด threshold ตรงศูนย์กลาง + อนุญาตรัศมีกว้างสุด
    dict(dp=1.2, param1=60, param2=20, minRadius=6,  maxRadius=85),
]

def adjust_gamma(img, gamma=1.2):
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.array([((i / 255.0) ** inv) * 255 for i in range(256)])).astype("uint8")
    return cv2.LUT(img, table)

def auto_roi(frame, name_det_result=None, default_band=(0.20, 0.85)):
    h = frame.shape[0]
    if name_det_result is not None:
        boxes = name_det_result[0].boxes.xyxy.cpu().numpy().astype(int)
        if boxes.shape[0] > 0:
            ys = []
            for (x1, y1, x2, y2) in boxes:
                ys += [y1, y2]
            top = max(0, min(ys) - int(0.10 * h))
            bot = min(h, max(ys) + int(0.10 * h))
            return top, bot
    # ถ้ายังไม่มีชื่อ ให้ใช้แถบกว้างๆ ไปก่อน
    t = int(h * default_band[0]); b = int(h * default_band[1])
    return t, b

def find_circles_multi(gray_roi):
    # ลองหลายเซตพารามิเตอร์จนกว่าจะเจอ
    for ps in CIRCLE_PARAM_SETS:
        circles = cv2.HoughCircles(
            gray_roi, cv2.HOUGH_GRADIENT,
            dp=ps["dp"], minDist=CIRCLE_MIN_DIST,
            param1=ps["param1"], param2=ps["param2"],
            minRadius=ps["minRadius"], maxRadius=ps["maxRadius"]
        )
        if circles is not None:
            yield np.uint16(np.around(circles[0, :]))

# ============================
def is_far_enough(new_center, centers, min_dist=50):
    new_center = np.array(new_center, dtype=np.float32)
    for c in centers:
        c = np.array(c, dtype=np.float32)
        dist = np.linalg.norm(new_center - c)
        if dist < min_dist:
            return False
    return True

# Load models
yolo_object_model = YOLO(YOLO_OBJECT_MODEL_PATH)
yolo_name_model = YOLO(YOLO_NAME_MODEL_PATH)
sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT).to("cuda")
predictor = SamPredictor(sam)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_count = 0
saved_frame_count = 0
fixed_centers = []
label_to_center = {}
hit_center_labels = set()
done_detecting = False
roi_top = height // 3
roi_bottom = 2 * height // 3
frame17_mask = None
frame18_mask = None
explosion_frame = None
explosion_detected = False
frame17_image = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    print(f"🔍 Frame {frame_count}")
    frame_copy = frame.copy()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not done_detecting:
        # --- เตรียมภาพ ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ภาพมืด/คอนทราสต์ต่ำ → ใช้ CLAHE + gamma เล็กน้อยให้สว่างขึ้น
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = adjust_gamma(gray, gamma=1.2)

        # (ลอง detect ชื่อแบบเบาๆ เพื่อช่วยกำหนด ROI)
        name_detections_try = yolo_name_model.predict(
            source=frame, conf=0.25, show=False, save=False, stream=False, verbose=False
        )

        # ROI อัตโนมัติ (อิงชื่อถ้ามี ไม่งั้นใช้แถบกว้าง 20%-85%)
        roi_top, roi_bottom = auto_roi(frame, name_detections_try, default_band=(0.20, 0.85))
        gray_roi = cv2.GaussianBlur(gray[roi_top:roi_bottom, :], (9, 9), sigmaX=2, sigmaY=2)

        # --- HoughCircles หลายพารามิเตอร์ ---
        found_this_frame = []
        for cset in find_circles_multi(gray_roi):
            for c in cset:
                center = (int(c[0]), int(c[1]) + roi_top)  # ใส่ offset กลับเข้าไป
                if is_far_enough(center, fixed_centers, min_dist=45):
                    found_this_frame.append(center)

            # ถ้าพบเยอะในพาสเดียวก็พอ
            if len(found_this_frame) + len(fixed_centers) >= 3:
                break

        # สะสมจุดที่ใหม่พอ
        for center in found_this_frame:
            fixed_centers.append(center)
            print(f"[✓] Frame {frame_count}: Added center {center}")
        if not found_this_frame:
            # log ดูเพื่อดีบักเวลาเจอแต่ซ้ำ
            print(f"⚠️ No NEW circles accepted at frame {frame_count} (kept={len(fixed_centers)})")

        # --- เงื่อนไขเข้าเฟสจับชื่อ + จับคู่กับวง ---
        if len(fixed_centers) >= 3:
            print("🔍 Trying to detect names (finalize mapping)...")
            name_detections = yolo_name_model.predict(
                source=frame, conf=0.30, show=False, save=False, stream=False, verbose=False
            )
            boxes = name_detections[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = name_detections[0].boxes.cls.cpu().numpy().astype(int)
            classnames = name_detections[0].names

            label_centers = []
            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                label = classnames[cls_id]
                if label in ['FR1', 'FR2', 'RE3']:
                    label_centers.append((label, (cx, cy)))

            # จับคู่ “ชื่อ” เข้ากับ “วง” ที่ใกล้ที่สุด
            temp_map = {}
            for label, pt in label_centers:
                best_center = min(fixed_centers, key=lambda c: np.linalg.norm(np.array(pt) - np.array(c)))
                temp_map[label] = best_center

            if all(l in temp_map for l in ['FR1', 'FR2', 'RE3']):
                label_to_center = temp_map
                done_detecting = True
                print("\n✅ Completed detection of circles and names.\n")

        # --- Fallback: ถ้าหาหลายเฟรมแล้วยังไม่ครบ 3 วง → ใช้ตำแหน่งชื่อแทน ---
        if (not done_detecting) and frame_count >= MAX_FRAMES_FOR_CIRCLE_SEARCH:
            print("🛟 Fallback: using name boxes as centers (circles insufficient).")
            boxes = name_detections_try[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = name_detections_try[0].boxes.cls.cpu().numpy().astype(int)
            classnames = name_detections_try[0].names

            temp_map = {}
            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = box
                label = classnames[cls_id]
                if label in ['FR1', 'FR2', 'RE3']:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    temp_map[label] = (cx, cy)
            if all(l in temp_map for l in ['FR1', 'FR2', 'RE3']):
                # ใช้ศูนย์กลางจากชื่อเป็น “ศูนย์กลางแทนวง”
                label_to_center = temp_map
                fixed_centers = list(temp_map.values())
                done_detecting = True
                print("\n✅ Fallback succeeded: using label centers for FR1/FR2/RE3.\n")
            else:
                print("❌ Fallback failed: not enough labels detected yet.")

        # วาดเพื่อดู
        for center in fixed_centers:
            cv2.circle(frame_copy, center, 30, (0, 255, 0), 2)
            cv2.circle(frame_copy, center, 5, (0, 0, 255), -1)

        out.write(frame_copy)
        continue

    for label, center in label_to_center.items():
        cv2.circle(frame_copy, center, 30, (0, 255, 0), 2)
        cv2.circle(frame_copy, center, 5, (0, 0, 255), -1)
        cv2.putText(frame_copy, label, (center[0] - 15, center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    yolo_results = yolo_object_model.predict(
        source=frame, conf=CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
    )
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
    predictor.set_image(image_rgb)
    frame_for_saving = frame_copy.copy()

    for box in boxes:
        x1, y1, x2, y2 = box
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = predictor.predict(
            box=input_box[None, :], multimask_output=True
        )
        best_mask = masks[np.argmax(scores)]

        if frame_count == 17:
            frame17_mask = best_mask.copy()
            frame17_image = frame_for_saving.copy()
            print("📌 Stored mask for frame 17")
        elif frame_count == 18:
            frame18_mask = best_mask.copy()
            print("📌 Stored mask for frame 18")
            if frame17_mask is not None:
                diff = np.logical_xor(frame17_mask, frame18_mask).astype(np.uint8)
                motion_score = np.sum(diff)
                print(f"🧮 Motion score between frame 17 & 18: {motion_score}")
                if motion_score > MOTION_THRESHOLD:
                    explosion_frame = 18
                    explosion_img = frame_for_saving
                    print("💥 Explosion detected at frame 18")
                else:
                    explosion_frame = 17
                    explosion_img = frame17_image
                    print("💥 Explosion detected at frame 17")
                screenshot_path = os.path.join(SCREENSHOT_DIR, f"Explosion_frame{explosion_frame}_{VIDEO_NAME}.png")
                cv2.imwrite(screenshot_path, explosion_img)
                explosion_detected = True
                print(f"💾 Saved explosion screenshot → {screenshot_path}")

        for label, center in label_to_center.items():
            if label in hit_center_labels:
                continue
            cx, cy = center
            if 0 <= cy < best_mask.shape[0] and 0 <= cx < best_mask.shape[1]:
                if best_mask[cy, cx]:
                    screenshot_path = os.path.join(SCREENSHOT_DIR, f"{label}_frame{frame_count}_{VIDEO_NAME}.png")
                    cv2.imwrite(screenshot_path, frame_for_saving)
                    saved_frame_count += 1
                    hit_center_labels.add(label)
                    print(f"💾 Saved frame at [{label}] (frame {frame_count}) → {screenshot_path}")

        green_mask = np.zeros_like(frame, dtype=np.uint8)
        green_mask[best_mask] = (0, 255, 0)
        frame_copy = cv2.addWeighted(frame_copy, 1.0, green_mask, 0.5, 0)

    out.write(frame_copy)

cap.release()
out.release()

cap_analysis = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
mask_area_data = []

# กำหนดช่วงเฟรมที่ต้องการวิเคราะห์
START_FRAME = 109
END_FRAME = 121

# เลื่อนไปยังเฟรมเริ่มต้น
cap_analysis.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME - 1) # -1 เพราะ CAP_PROP_POS_FRAMES เริ่มนับจาก 0

while cap_analysis.isOpened():
    ret, frame = cap_analysis.read()
    if not ret:
        break
    frame_count = int(cap_analysis.get(cv2.CAP_PROP_POS_FRAMES)) # อัปเดต frame_count ตามเฟรมปัจจุบัน

    # หยุดการทำงานเมื่อถึงเฟรมสิ้นสุดที่กำหนด
    if frame_count > END_FRAME:
        break

    print(f"🔄 Analyzing frame {frame_count} for full deployment...")

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    yolo_results = yolo_object_model.predict(
        source=frame, conf=CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
    )
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)

    if len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = box
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            box=np.array([x1, y1, x2, y2])[None, :], multimask_output=True
        )
        best_mask = masks[np.argmax(scores)]
        area = np.sum(best_mask)
        mask_area_data.append((frame_count, area))

cap_analysis.release()

if len(mask_area_data) > 0:
    areas = np.array([area for _, area in mask_area_data])
    frames = [frame_num for frame_num, _ in mask_area_data]

    # ตรวจสอบให้แน่ใจว่ามีข้อมูลเพียงพอสำหรับการทำ Smoothing
    if len(areas) >= 3: # ต้องมีอย่างน้อย 3 จุดสำหรับ uniform_filter1d(size=3)
        smoothed_areas = uniform_filter1d(areas, size=SMOOTH_SIZE)
    else:
        # ถ้าข้อมูลน้อยเกินไป ให้ใช้พื้นที่เดิมโดยไม่มีการ smoothing
        smoothed_areas = areas
        print("⚠️ Not enough data points for effective smoothing. Using raw area data for plateau detection.")

    peak_index = np.argmax(smoothed_areas)
    plateau_frame = frames[peak_index]
    
    # กำหนด threshold ให้ยืดหยุ่นขึ้นเล็กน้อย
    # ถ้าพื้นที่สูงสุดเป็น 0 (กรณีไม่มี detection ที่มี mask) ให้ threshold เป็น 0 ด้วย
    threshold = smoothed_areas[peak_index] * PLATEAU_ALPHA if smoothed_areas[peak_index] > 0 else 0

    # หาจุดที่พื้นที่เริ่มลดลงจาก plateau (หรือคงที่ในระดับสูง)
    for i in range(peak_index, len(smoothed_areas)):
        if smoothed_areas[i] < threshold:
            break
        plateau_frame = frames[i]

    print(f"\n✨ Full Deployment Detected at Frame (within 109-121): {plateau_frame} (Smoothed Area: {smoothed_areas[peak_index]})")

    cap_final = cv2.VideoCapture(VIDEO_PATH)
    cap_final.set(cv2.CAP_PROP_POS_FRAMES, plateau_frame - 1)
    ret_final, frame_final = cap_final.read()
    if ret_final:
        screenshot_path = os.path.join(
            SCREENSHOT_DIR, f"Airbag_Full_Deployment_Smoothed_frame{plateau_frame}_{VIDEO_NAME}.png")
        cv2.imwrite(screenshot_path, frame_final)
        print(f"💾 Captured full deployment screenshot → {screenshot_path}")
    else:
        print("❌ Could not read final frame for full deployment capture.")
    cap_final.release()
else:
    print("\n❗ No airbag mask data collected within the specified frame range (109-121).")


print("\n=== Final Fixed Centers with Labels ===")
for label, c in label_to_center.items():
    print(f"{label}: ({int(c[0])}, {int(c[1])})")
print(f"\n✅ Process completed. Total saved frames: {saved_frame_count}")