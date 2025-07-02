import torch
import cv2
import numpy as np
import os
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

# ======= CONFIG ============
YOLO_MODEL_PATH = r"Model\best.pt"
SAM_CHECKPOINT = r"Model\sam_vit_b_01ec64.pth"
SAM_TYPE = "vit_b"
VIDEO_PATH = r"Video\N1WB-E042D94-AEDYACB25141110371_23_Side.avi"
OUTPUT_PATH = r"Output\segment.mp4"
MASK_VIDEO_PATH = r"Output\Mask\mask_video.mp4"
CONFIDENCE_THRESHOLD = 0.5
DEVICE = "cpu"  # "cuda" ถ้าใช้ GPU ได้
# ============================

# Load models
yolo_model = YOLO(YOLO_MODEL_PATH)
sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT).to(DEVICE)
predictor = SamPredictor(sam)

# Read input video
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
out_mask = cv2.VideoWriter(MASK_VIDEO_PATH, fourcc, fps, (width, height), isColor=False)

frame_count = 0
os.makedirs("Video", exist_ok=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"🧠 Processing frame {frame_count}...")

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        show=False,
        save=False,
        stream=False,
        verbose=False
    )

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    predictor.set_image(image_rgb)

    overlay = frame.copy()
    full_mask = np.zeros((height, width), dtype=np.uint8)

    for box in boxes:
        x1, y1, x2, y2 = box
        input_box = np.array([x1, y1, x2, y2])

        masks, scores, _ = predictor.predict(
            box=input_box[None, :],
            multimask_output=True
        )
        best_mask = masks[np.argmax(scores)]

        green_mask = np.zeros_like(frame, dtype=np.uint8)
        green_mask[best_mask] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 1.0, green_mask, 0.5, 0)

        # เก็บรวม Mask
        full_mask[best_mask] = 255

    out_video.write(overlay)
    out_mask.write(full_mask)

cap.release()
out_video.release()
out_mask.release()
print("✅ Saved overlay video:", OUTPUT_PATH)
print("✅ Saved mask video:", MASK_VIDEO_PATH)
