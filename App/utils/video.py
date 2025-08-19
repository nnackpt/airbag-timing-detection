import os, cv2
from typing import Tuple
from App.config import settings

def ensure_dirs(video_name: str) -> tuple[str, str]:
    out_dir = settings.OUTPUT_ROOT
    shot_dir = os.path.join(settings.SCREENSHOT_ROOT, video_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(shot_dir, exist_ok=True)
    return out_dir, shot_dir

def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, width, height, fps

def writer(output_path: str, fps: float, wh: Tuple[int, int]):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, wh)