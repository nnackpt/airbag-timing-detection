import cv2
import numpy as np
import torch
import os
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
 
# ======= CONFIG ============
YOLO_MODEL_PATH = r"runs\detect\sam\8m50e\weights\best.pt"
SAM_CHECKPOINT = r"sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"
VIDEO_PATH = r"rrr.avi"
VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0] 
OUTPUT_PATH = fr"Output\{VIDEO_NAME}_Timing_Detection.mp4"
SCREENSHOT_DIR = os.path.join("Output", "Screenshots", VIDEO_NAME)
CONFIDENCE_THRESHOLD = 0.5
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
# ============================
 
def is_far_enough(new_center, centers, min_dist=50):
    for c in centers:
        dist = np.linalg.norm(np.array(new_center) - np.array(c))
        print(f"    ↪️ Checking distance: {dist:.2f}")
        if dist < min_dist:
            return False
    return True
 
# Load models
yolo_model = YOLO(YOLO_MODEL_PATH)
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
hit_center_indexes = set()
done_detecting = False
 
roi_top = height // 3
roi_bottom = 2 * height // 3
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    frame_count += 1
    print(f"🔍 Frame {frame_count}")
    frame_copy = frame.copy()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Step 1: Detect 3 unique circles
    if not done_detecting:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        roi = gray[roi_top:roi_bottom, :]
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=60,
            param1=100,
            param2=40,
            minRadius=5,
            maxRadius=30
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                center = (int(c[0]), int(c[1] + roi_top))
                if is_far_enough(center, fixed_centers, min_dist=20):
                    fixed_centers.append(center)
                    print(f"[✓] Frame {frame_count}: Added center {center}")
                if len(fixed_centers) >= 3:
                    done_detecting = True
                    print("\n✅ Completed detection of 3 unique circles.\n")
                    break
 
        # Draw centers (while collecting)
        for center in fixed_centers:
            cv2.circle(frame_copy, center, 30, (0, 255, 0), 2)
            cv2.circle(frame_copy, center, 5, (0, 0, 255), -1)
 
        out.write(frame_copy)
        continue
 
    # Step 2: Draw fixed circles
    for center in fixed_centers:
        cv2.circle(frame_copy, center, 30, (0, 255, 0), 2)
        cv2.circle(frame_copy, center, 5, (0, 0, 255), -1)

    sorted_centers = sorted([(i, c) for i, c in enumerate(fixed_centers)], key=lambda x: x[1][0])
    labels = ['L', 'C', 'R']
    for idx, (i, center) in enumerate(sorted_centers):
        cv2.circle(frame_copy, center, 30, (0, 255, 0), 2)
        cv2.circle(frame_copy, center, 5, (0, 0, 255), -1)
        cv2.putText(frame_copy, labels[idx], (center[0] - 10, center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Step 3: YOLO
    yolo_results = yolo_model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        show=False,
        save=False,
        stream=False,
        verbose=False
    )
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
 
    # Step 4: SAM
    predictor.set_image(image_rgb)
    frame_for_saving = frame_copy.copy()  # ⭐ สำรองไว้ก่อนวาด overlay
 
    for box in boxes:
        x1, y1, x2, y2 = box
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = predictor.predict(
            box=input_box[None, :],
            multimask_output=True
        )
        best_mask = masks[np.argmax(scores)]
 
        # Step 5: Check hit per center point
        frame_before_overlay = frame_copy.copy()

        # SAM predict
        predictor.set_image(image_rgb)

        # กลุ่มศูนย์กลางจากซ้าย-กลาง-ขวา
        sorted_centers = sorted([(i, c) for i, c in enumerate(fixed_centers)], key=lambda x: x[1][0])
        labels = ['left', 'center', 'right']
        center_label_map = {i: labels[idx] for idx, (i, _) in enumerate(sorted_centers)}

        # สร้าง Mask ทับ frame ทีละ box
        for box in boxes:
            x1, y1, x2, y2 = box
            input_box = np.array([x1, y1, x2, y2])
            masks, scores, _ = predictor.predict(
                box=input_box[None, :],
                multimask_output=True
            )
            best_mask = masks[np.argmax(scores)]

            # ✅ ตรวจจุดชนก่อนวาด overlay
            for idx, center in enumerate(fixed_centers):
                cx, cy = center
                if 0 <= cy < best_mask.shape[0] and 0 <= cx < best_mask.shape[1]:
                    if best_mask[cy, cx] and idx not in hit_center_indexes:
                        label = center_label_map[idx]
                        screenshot_path = os.path.join(
                            SCREENSHOT_DIR,
                            f"{label}_frame{frame_count}_{VIDEO_NAME}.png"
                        )
                        cv2.imwrite(screenshot_path, frame_before_overlay)  # ✅ ใช้ภาพก่อน overlay จริงๆ
                        saved_frame_count += 1
                        hit_center_indexes.add(idx)
                        print(f"💾 Saved frame at [{label}] (frame {frame_count}) → {screenshot_path}")

            # 🖌️ วาด segmentation mask ลง frame
            green_mask = np.zeros_like(frame, dtype=np.uint8)
            green_mask[best_mask] = (0, 255, 0)
            frame_copy = cv2.addWeighted(frame_copy, 1.0, green_mask, 0.5, 0)
 
    out.write(frame_copy)
 
cap.release()
out.release()
 
print("\n=== Final Fixed Centers (3 Unique Circles) ===")
for i, c in enumerate(fixed_centers):
    print(f"Point {i+1}: {c}")
print(f"\n✅ Process completed. Total saved frames: {saved_frame_count}")