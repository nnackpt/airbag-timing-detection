# CircleCheck.py

import cv2
import numpy as np
import os

# ===== CONFIG =====
VIDEO_PATH = r'Output\segment.mp4'
MASK_VIDEO_PATH = r'Output\Mask\mask_video.mp4'
OUTPUT_VIDEO = r'Output\segment_with_circles.mp4'
SCREENSHOT_DIR = 'Screenshots'
RED_TOUCH_RADIUS = 5  # px
# ===================

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
mask_cap = cv2.VideoCapture(MASK_VIDEO_PATH)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_id = 0
touched_frames = set()

while cap.isOpened() and mask_cap.isOpened():
    ret, frame = cap.read()
    ret_mask, mask_frame = mask_cap.read()

    if not ret or not ret_mask:
        break

    frame_id += 1
    print(f"🔍 Frame {frame_id}")

    # Convert mask frame to grayscale
    if len(mask_frame.shape) == 3:
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_frame

    mask_binary = mask_gray > 128

    # Detect circles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=30
    )

    hit = False
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cx, cy, r = i[0], i[1], i[2]

            # Draw circles
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 3)

            # Check hit
            x1 = max(0, cx - RED_TOUCH_RADIUS)
            y1 = max(0, cy - RED_TOUCH_RADIUS)
            x2 = min(frame_width, cx + RED_TOUCH_RADIUS)
            y2 = min(frame_height, cy + RED_TOUCH_RADIUS)

            region = mask_binary[y1:y2, x1:x2]
            if np.any(region):
                hit = True
                if frame_id not in touched_frames:
                    screenshot_path = os.path.join(SCREENSHOT_DIR, f"hit_{frame_id:05d}.jpg")
                    cv2.imwrite(screenshot_path, frame)
                    touched_frames.add(frame_id)
                    print(f"📸 Screenshot saved: {screenshot_path}")

    out.write(frame)

cap.release()
mask_cap.release()
out.release()
print("✅ Final video saved:", OUTPUT_VIDEO)
