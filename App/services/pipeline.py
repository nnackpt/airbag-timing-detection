from typing import Dict, Tuple, Optional, List
import os, cv2, numpy as np
from scipy.ndimage import uniform_filter1d
from App.config import settings
from App.services.models import get_models
from App.utils.video import open_video, writer, ensure_dirs
from App.utils.geometry import is_far_enough

CIRCLE_MIN_DIST = 45
CIRCLE_PARAM_SETS = [
    dict(dp=1.2, param1=80, param2=30, minRadius=8, maxRadius=60),
    dict(dp=1.0, param1=70, param2=24, minRadius=8, maxRadius=72),
    dict(dp=1.2, param1=60, param2=20, minRadius=6, maxRadius=85),
]
MAX_FRAMES_FOR_CIRCLE_SEARCH = 80

def _adjust_gamma(img, gamma=1.2):
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.array([((i / 255.0) ** inv) * 255 for i in range(256)])).astype("uint8")
    return cv2.LUT(img, table)

def _auto_roi(frame, name_res=None, default_band=(0.20, 0.85)):
    h = frame.shape[0]
    if name_res is not None:
        boxes = name_res[0].boxes.xyxy.cpu().numpy().astype(int)
        if boxes.shape[0] > 0:
            ys = []
            for (x1, y1, x2, y2) in boxes:
                ys += [y1, y2]
            top = max(9, min(ys) - int(0.10 * h))
            bot = min(h, max(ys) + int(0.10 * h))
            return top, bot
    t = int(h * default_band[0]); b = int(h * default_band[1])
    return t, b

def _find_circles(gray_roi):
    for ps in CIRCLE_PARAM_SETS:
        circles = cv2.HoughCircles(
            gray_roi, cv2.HOUGH_GRADIENT,
            dp=ps["dp"], minDist=CIRCLE_MIN_DIST,
            param1=ps["param1"], param2=ps["param2"],
            minRadius=ps["minRadius"], maxRadius=ps["maxRadius"]
        )
        if circles is not None:
            yield np.uint16(np.around(circles[0, :]))
            
def detect_centers_and_labels(video_path: str) -> Tuple[Dict[str, Tuple[int, int]], List[Tuple[int, int]]]:
    M = get_models()
    cap, width, height, fps = open_video(video_path)
    
    fixed_centers: List[Tuple[int, int]] = []
    label_to_center: Dict[str, Tuple[int, int]] = {}
    done = False
    frame_idx = 0
    
    while cap.isOpened() and not done:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = _adjust_gamma(gray, 1.2)
        
        # Light name detect to guide ROI
        name_try = M.yolo_name.predict(source=frame, conf=0.25, stream=False, verbose=False)
        top, bot = _auto_roi(frame, name_try, default_band=(0.20, 0.85))
        gray_roi = cv2.GaussianBlur(gray[top:bot, :], (9, 9), sigmaX=2, sigmaY=2)
        
        found_this = []
        for cset in _find_circles(gray_roi):
            for c in cset:
                center = (int(c[0]), int(c[1]) + top)
                if is_far_enough(center, fixed_centers, min_dist=45):
                    found_this.append(center)
            if len(found_this) + len(fixed_centers) >= 3:
                break
        for center in found_this:
            fixed_centers.append(center)
            
        if len(fixed_centers) >= 3:
            det = M.yolo_name.predict(source=frame, conf=0.30, stream=False, verbose=False)
            boxes = det[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = det[0].boxes.cls.cpu().numpy().astype(int)
            names = det[0].names
            label_pts = []
            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = box
                label = names[cls_id]
                if label in ["FR1", "FR2", "RE3"]:
                    cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                    label_pts.append((label,(cx, cy)))
            tmp = {}
            for label, pt in label_pts:
                best = min(fixed_centers, key=lambda c: np.linalg.norm(np.array(pt)-np.array(c)))
                tmp[label] = best
            if all(k in tmp for k in ["FR1", "FR2", "RE3"]):
                label_to_center = tmp
                done = True
                
        if not done and frame_idx >= MAX_FRAMES_FOR_CIRCLE_SEARCH:
            boxes = name_try[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = name_try[0].boxes.cls.cpu().numpy().astype(int)
            names = name_try[0].names
            tmp = {}
            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = box
                label = names[cls_id]
                if label in ["FR1", "FR2", "RE3"]:
                    cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                    tmp[label] = (cx, cy)
            if all(k in tmp for k in ["FR1", "FR2", "RE3"]):
                label_to_center = tmp
                fixed_centers = list(tmp.values())
                done = True
    
    cap.release()
    return label_to_center, fixed_centers

def detect_explosion_17_18(video_path: str, motion_threshold: int | None = None):
    M = get_models()
    mot_th = motion_threshold or settings.MOTION_THRESHOLD
    
    cap, w, h, fps = open_video(video_path)
    frame17_mask = None
    frame17_img = None
    frame18_mask = None
    explosion_frame = None
    
    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx not in (17, 18):
            continue
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = M.yolo_object.predict(source=frame, conf=settings.CONFIDENCE_THRESHOLD, stream=False, verbose=False)
        boxes = det[0].boxes.xyxy.cpu().numpy().astype(int)
        if len(boxes) == 0:
            continue
        x1, y1, x2, y2 = boxes[0]
        M.sam_predictor.set_image(image_rgb)
        masks, scores, _ = M.sam_predictor.predict(box=np.array([x1, y1, x2, y2])[None, :], multimask_output=True)
        best_mask = masks[np.argmax(scores)]
        if frame_idx == 17:
            frame17_mask = best_mask.copy(); frame17_img = frame.copy()
        else :
            frame18_mask = best_mask.copy()
    cap.release()
    
    if frame17_mask is None or frame18_mask is None:
        return 17, None # Fallback
    diff = np.logical_xor(frame17_mask, frame18_mask).astype(np.uint8)
    motion_score = int(np.sum(diff))
    explosion_frame = 18 if motion_score > mot_th else 17
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    _, shot_dir = ensure_dirs(video_name)
    shot_path = os.path.join(shot_dir, f"Explosion_frame{explosion_frame}_{video_name}.png")
    cv2.imwrite(shot_path, frame17_img if explosion_frame == 17 else frame18_mask.astype(np.uint8) * 255)
    return explosion_frame, shot_path

def analyze_full_deployment(video_path: str, window: Tuple[int, int], *,
                            confidence_threshold: float | None = None,
                            smooth_size: int | None = None,
                            plateau_alpha: float | None = None):
    M = get_models()
    conf = confidence_threshold or settings.CONFIDENCE_THRESHOLD
    k = smooth_size or settings.SMOOTH_SIZE
    alpha = plateau_alpha or settings.PLATEAU_ALPHA
    
    cap = cv2.VideoCapture(video_path)
    start, end = window
    cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
    
    frame_nums: List[int] = []
    areas: List[int] = []
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cur > end:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = M.yolo_object.predict(source=frame, conf=conf, stream=False, verbose=False)
        boxes = det[0].boxes.xyxy.cpu().numpy().astype(int)
        if len(boxes) == 0:
            continue
        x1, y1, x2, y2 = boxes[0]
        M.sam_predictor.set_image(image_rgb)
        masks, scores, _ = M.sam_predictor.predict(box=np.array([x1, y1, x2, y2])[None, :], multimask_output=True)
        best_mask = masks[np.argmax(scores)]
        areas.append(int(np.sum(best_mask)))
        frame_nums.append(cur)
        
    cap.release()
    
    if not areas:
        return None, None, None
    
    arr = np.array(areas)
    if len(arr) >= 3:
        smoothed = uniform_filter1d(arr, size=k)
    else :
        smoothed = arr
    peak_idx = int(np.argmax(smoothed))
    threshold = smoothed[peak_idx] * alpha if smoothed[peak_idx] > 0 else 0
    plateau_frame = frame_nums[peak_idx]
    for i in range(peak_idx, len(smoothed)):
        if smoothed[i] < threshold:
            break
        plateau_frame = frame_nums[i]
        
    # Screenshot
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    _, shot_dir = ensure_dirs(video_name)
    cap2 = cv2.VideoCapture(video_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, plateau_frame - 1)
    ok, frame_final = cap2.read()
    cap2.release()
    shot_path = None
    if ok:
        shot_path = os.path.join(shot_dir, f"Airbag_Full_Deployment_frame_{plateau_frame}_{video_name}.png")
        cv2.imwrite(shot_path, frame_final)
    return plateau_frame, int(smoothed[peak_idx]), shot_path

def run_full_pipeline(video_path: str,
                      *,
                      condition: Optional[str] = None,
                      window: Optional[Tuple[int, int]] = None,
                      save_video: bool = True,
                      params: dict | None = None):
    # Params: Override thresholds
    if params is None:
        params = {}
        
    label_to_center, fixed = detect_centers_and_labels(video_path)
    explosion_frame, exp_path = detect_explosion_17_18(video_path, params.get("motion_threshold"))
    
    if window is None and condition is not None:
        window = settings.CONDITION_WINDOWS[condition]
    if window is None:
        window = settings.CONDITION_WINDOWS["room"]
        
    plateau_frame, smoothed_peak, shot = analyze_full_deployment(
        video_path, window,
        confidence_threshold=params.get("confidence_threshold"),
        smooth_size=params.get("smooth_size"),
        plateau_alpha=params.get("plateau_alpha")
    )
    
    out_path = None
    if save_video:
        cap, w, h, fps = open_video(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir, _ = ensure_dirs(video_name)
        out_path = os.path.join(out_dir, f"{video_name}_Timing_Detection.mp4")
        vw = writer(out_path, fps, (w,h))
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vw.write(frame)
        cap.release(); vw.release()
        
    return {
        "labeled_centers": label_to_center or None,
        "explosion": {
            "explosion_frame": explosion_frame,
            "screenshot_path": exp_path
        },
        "full_deployment": {
            "plateau_frame": plateau_frame,
            "smoothed_peak": smoothed_peak,
            "screenshot_path": shot
        },
        "output_video_path": out_path
    }