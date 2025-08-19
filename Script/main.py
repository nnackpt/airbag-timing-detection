from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import asyncio
from pathlib import Path
import uuid
from pydantic import BaseModel
import subprocess
import logging
from PIL import Image
import cv2
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from io import BytesIO
from transformers import TextIteratorStreamer
from threading import Thread
import re
from enum import Enum

# กำหนด logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "Uploads"
OUTPUT_DIR = BASE_DIR / "Outputs"
SCREENSHOT_DIR = BASE_DIR / "Screenshots"
MODEL_DIR = BASE_DIR / "Model"
DB_PATH = BASE_DIR / "Airbag_detection.db"

# สร้างโฟลเดอร์ที่จำเป็น
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, SCREENSHOT_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model paths
YOLO_OBJECT_MODEL_PATH = MODEL_DIR / "YOLO_OBJECT.pt"
YOLO_NAME_MODEL_PATH = MODEL_DIR / "YOLO_NAME.pt"
SAM_CHECKPOINT = MODEL_DIR / "sam_vit_h_4b8939.pth"
OCR_MODEL_PATH = MODEL_DIR / "ocr"

# ===== MODELS =====
class TemperatureType(str, Enum):
    ROOM = "room"
    HOT = "hot"
    COLD = "cold"

class VideoUploadRequest(BaseModel):
    temperature_type: TemperatureType = TemperatureType.ROOM

class VideoUploadResponse(BaseModel):
    task_id: str
    message: str
    video_filename: str
    temperature_type: str
    
class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    progress: int  # 0-100
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    temperature_type: Optional[str] = None

class VideoRecord(BaseModel):
    id: int
    task_id: str
    original_filename: str
    video_filename: str
    status: str
    progress: int
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_video_path: Optional[str] = None
    screenshots: List[str] = []
    temperature_type: Optional[str] = None

class DetectionResult(BaseModel):
    task_id: str
    video_filename: str
    explosion_frame: Optional[int] = None
    full_deployment_frame: Optional[int] = None
    detected_labels: List[str] = []
    screenshots: List[str] = []
    processing_time: float
    ocr_results: Dict[str, str] = {}

# ===== DATABASE =====
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Videos table (เพิ่มคอลัมน์ temperature_type)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT UNIQUE NOT NULL,
            original_filename TEXT NOT NULL,
            video_filename TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            message TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME,
            output_video_path TEXT,
            screenshots TEXT,  -- JSON array of screenshot paths
            ocr_results TEXT,  -- JSON object of OCR results
            temperature_type TEXT DEFAULT 'room'  -- เพิ่มคอลัมน์ใหม่
        )
    ''')
    
    # เพิ่มคอลัมน์ temperature_type หากไม่มี (สำหรับ DB เดิม)
    try:
        cursor.execute('ALTER TABLE videos ADD COLUMN temperature_type TEXT DEFAULT "room"')
        conn.commit()
    except sqlite3.OperationalError:
        pass  # คอลัมน์มีอยู่แล้ว
    
    # Processing logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            log_level TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES videos (task_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ===== OCR CLASS =====
class OCR:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self.processor = AutoProcessor.from_pretrained(
                OCR_MODEL_PATH,
                trust_remote_code=True,
                use_fast=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                OCR_MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.float32
            ).to(self.device).eval()
        except Exception as e:
            logger.error(f"Failed to load OCR model: {e}")
            self.processor = None
            self.model = None

    def extract_ms_time(self, image: Image.Image, max_new_tokens: int = 32) -> str:
        if self.model is None or image is None:
            return "OCR model not available"
        
        try:
            query = (
                "Read the time (in milliseconds) shown at the top-left corner of the image. "
                "Return exactly in this format: ms=13.6 (The time should be a number only, without '+' or '-' sign)"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": query}
                    ]
                }
            ]

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)

            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 30,
                "repetition_penalty": 1.1
            }

            Thread(target=self.model.generate, kwargs=generation_kwargs).start()
            output_text = "".join(token for token in streamer).strip().replace("<|im_end|>", "").strip()

            # 🔍 Regex ดึงค่า ms
            match = re.search(r"ms\s*=\s*([+-]?\d+\.?\d*)", output_text, re.IGNORECASE)
            if match:
                ms_val = match.group(1).lstrip("+")  # ตัด '+' ออก
                return ms_val
            
            return f"Unable to extract ms from output: {output_text}"
        
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return "OCR extraction failed"

# ===== AIRBAG DETECTION CLASS =====
class AirbagDetector:
    def __init__(self):
        self.yolo_object_model = None
        self.yolo_name_model = None
        self.sam = None
        self.predictor = None
        self.ocr = OCR()
        self.load_models()
    
    def load_models(self):
        """Load all required models"""
        try:
            if YOLO_OBJECT_MODEL_PATH.exists():
                self.yolo_object_model = YOLO(str(YOLO_OBJECT_MODEL_PATH))
                logger.info("YOLO Object model loaded successfully")
            else:
                logger.warning(f"YOLO Object model not found at {YOLO_OBJECT_MODEL_PATH}")
            
            if YOLO_NAME_MODEL_PATH.exists():
                self.yolo_name_model = YOLO(str(YOLO_NAME_MODEL_PATH))
                logger.info("YOLO Name model loaded successfully")
            else:
                logger.warning(f"YOLO Name model not found at {YOLO_NAME_MODEL_PATH}")
            
            if SAM_CHECKPOINT.exists():
                self.sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
                if torch.cuda.is_available():
                    self.sam.to("cuda")
                self.predictor = SamPredictor(self.sam)
                logger.info("SAM model loaded successfully")
            else:
                logger.warning(f"SAM model not found at {SAM_CHECKPOINT}")
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def is_far_enough(self, new_center, centers, min_dist=50):
        """Check if new center is far enough from existing centers"""
        new_center = np.array(new_center, dtype=np.float32)
        for c in centers:
            c = np.array(c, dtype=np.float32)
            dist = np.linalg.norm(new_center - c)
            if dist < min_dist:
                return False
        return True
    
    def get_temperature_frame_range(self, temperature_type: str):
        """Get frame range based on temperature type"""
        if temperature_type == "hot":
            return 102, 115
        elif temperature_type == "cold":
            return 112, 131
        else:  # room temperature (default)
            return 109, 121
    
    def process_video(self, video_path: str, task_id: str, temperature_type: str = "room", callback=None):
        """Process video for airbag detection with temperature selection"""
        try:
            if not self.yolo_object_model or not self.yolo_name_model or not self.sam:
                raise Exception("Models not loaded properly")
            
            video_name = Path(video_path).stem
            output_video_path = OUTPUT_DIR / f"{video_name}_Timing_Detection_{temperature_type}.mp4"
            screenshot_dir = SCREENSHOT_DIR / f"{video_name}_{temperature_type}"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Update status
            if callback:
                callback(task_id, "processing", 5, f"Opening video for {temperature_type} temperature analysis...")
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception("Could not open video file")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            
            # Processing variables (ตามไฟล์ Python หลัก)
            frame_count = 0
            saved_frame_count = 0
            saved_screenshots = []
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
            
            CONFIDENCE_THRESHOLD = 0.5
            MOTION_THRESHOLD = 1500
            
            if callback:
                callback(task_id, "processing", 10, "Starting frame processing...")
            
            # Process frames for circle and name detection
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                logger.info(f"Processing Frame {frame_count}/{total_frames}")
                
                # Update progress
                if callback and frame_count % 10 == 0:
                    progress = min(int((frame_count / total_frames) * 70), 70)
                    callback(task_id, "processing", progress, f"Processing frame {frame_count}/{total_frames}")
                
                frame_copy = frame.copy()
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Circle detection phase (ตามไฟล์ Python หลัก)
                if not done_detecting:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # เพิ่ม contrast ด้วย CLAHE สำหรับภาพมืด
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)

                    # ปรับความเนียน
                    gray_blurred = cv2.GaussianBlur(gray[roi_top:roi_bottom, :], (9, 9), sigmaX=2, sigmaY=2)

                    # ปรับค่าพารามิเตอร์ HoughCircles ใหม่ให้เสถียรกว่าเดิม
                    circles = cv2.HoughCircles(
                        gray_blurred,
                        cv2.HOUGH_GRADIENT,
                        dp=1.2,
                        minDist=60,               # ป้องกันจุดซ้ำ
                        param1=80,                # edge detector threshold
                        param2=30,                # circle center sensitivity
                        minRadius=10,
                        maxRadius=35
                    )

                    if circles is not None:
                        circles = np.uint16(np.around(circles[0, :]))
                        for c in circles:
                            center = (c[0], c[1] + roi_top)  # อย่าลืมบวกกลับ
                            if self.is_far_enough(center, fixed_centers, min_dist=60):
                                fixed_centers.append(center)
                                logger.info(f"[✓] Frame {frame_count}: Added center {center}")
                            else:
                                logger.info(f"⚠️ Center too close to existing: {center}")
                        
                        # Name detection when we have 3 circles
                        if len(fixed_centers) == 3:
                            logger.info("Trying to detect names...")
                            name_detections = self.yolo_name_model.predict(
                                source=frame, conf=0.4, show=False, save=False, stream=False, verbose=False
                            )
                            
                            if name_detections[0].boxes is not None:
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
                                
                                # Match labels to centers
                                temp_map = {}
                                for label, pt in label_centers:
                                    best_center = min(fixed_centers, key=lambda c: np.linalg.norm(np.array(pt) - np.array(c)))
                                    temp_map[label] = best_center
                                
                                if all(l in temp_map for l in ['FR1', 'FR2', 'RE3']):
                                    label_to_center = temp_map
                                    done_detecting = True
                                    logger.info("Completed detection of circles and names")
                        
                        # Draw circles on frame
                        for center in fixed_centers:
                            cv2.circle(frame_copy, center, 30, (0, 255, 0), 2)
                            cv2.circle(frame_copy, center, 5, (0, 0, 255), -1)
                        
                        out.write(frame_copy)
                        continue
                
                # Object detection phase (ตามไฟล์ Python หลัก)
                for label, center in label_to_center.items():
                    cv2.circle(frame_copy, center, 30, (0, 255, 0), 2)
                    cv2.circle(frame_copy, center, 5, (0, 0, 255), -1)
                    cv2.putText(frame_copy, label, (center[0] - 15, center[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # YOLO object detection
                yolo_results = self.yolo_object_model.predict(
                    source=frame, conf=CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
                )
                
                if yolo_results[0].boxes is not None:
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
                    self.predictor.set_image(image_rgb)
                    frame_for_saving = frame_copy.copy()
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        input_box = np.array([x1, y1, x2, y2])
                        masks, scores, _ = self.predictor.predict(
                            box=input_box[None, :], multimask_output=True
                        )
                        best_mask = masks[np.argmax(scores)]
                        
                        # Explosion detection (frame 17-18)
                        if frame_count == 17:
                            frame17_mask = best_mask.copy()
                            frame17_image = frame_for_saving.copy()
                            logger.info("Stored mask for frame 17")
                        elif frame_count == 18:
                            frame18_mask = best_mask.copy()
                            logger.info("Stored mask for frame 18")
                            
                            if frame17_mask is not None:
                                diff = np.logical_xor(frame17_mask, frame18_mask).astype(np.uint8)
                                motion_score = np.sum(diff)
                                logger.info(f"Motion score between frame 17 & 18: {motion_score}")
                                
                                if motion_score > MOTION_THRESHOLD:
                                    explosion_frame = 18
                                    explosion_img = frame_for_saving
                                    logger.info("Explosion detected at frame 18")
                                else:
                                    explosion_frame = 17
                                    explosion_img = frame17_image
                                    logger.info("Explosion detected at frame 17")
                                
                                screenshot_path = screenshot_dir / f"Explosion_frame{explosion_frame}_{video_name}.png"
                                cv2.imwrite(str(screenshot_path), explosion_img)
                                saved_screenshots.append(str(screenshot_path))
                                explosion_detected = True
                                logger.info(f"Saved explosion screenshot: {screenshot_path}")
                        
                        # Check if mask hits any center
                        for label, center in label_to_center.items():
                            if label in hit_center_labels:
                                continue
                            cx, cy = center
                            if 0 <= cy < best_mask.shape[0] and 0 <= cx < best_mask.shape[1]:
                                if best_mask[cy, cx]:
                                    screenshot_path = screenshot_dir / f"{label}_frame{frame_count}_{video_name}.png"
                                    cv2.imwrite(str(screenshot_path), frame_for_saving)
                                    saved_screenshots.append(str(screenshot_path))
                                    saved_frame_count += 1
                                    hit_center_labels.add(label)
                                    logger.info(f"Saved frame at [{label}] (frame {frame_count})")
                        
                        # Add green mask overlay
                        green_mask = np.zeros_like(frame, dtype=np.uint8)
                        green_mask[best_mask] = (0, 255, 0)
                        frame_copy = cv2.addWeighted(frame_copy, 1.0, green_mask, 0.5, 0)
                
                out.write(frame_copy)
            
            cap.release()
            out.release()
            
            # Full deployment detection (ตามไฟล์ Python หลัก)
            if callback:
                callback(task_id, "processing", 75, f"Analyzing full deployment for {temperature_type} temperature...")
            
            full_deployment_frame = self.analyze_full_deployment(video_path, video_name, screenshot_dir, temperature_type)
            
            # OCR processing
            if callback:
                callback(task_id, "processing", 90, "Running OCR on screenshots...")
            
            ocr_results = {}
            for screenshot_path in saved_screenshots:
                if any(label in Path(screenshot_path).name for label in ['FR1', 'FR2', 'RE3']):
                    try:
                        image = Image.open(screenshot_path).convert("RGB")
                        ocr_result = self.ocr.extract_ms_time(image)
                        ocr_results[Path(screenshot_path).name] = ocr_result
                    except Exception as e:
                        logger.error(f"OCR failed for {screenshot_path}: {e}")
                        ocr_results[Path(screenshot_path).name] = "OCR failed"
            
            # Final update
            if callback:
                callback(task_id, "completed", 100, "Processing completed successfully", 
                        str(output_video_path), saved_screenshots, ocr_results)
            
            return {
                "success": True,
                "output_video": str(output_video_path),
                "screenshots": saved_screenshots,
                "explosion_frame": explosion_frame,
                "full_deployment_frame": full_deployment_frame,
                "detected_labels": list(hit_center_labels),
                "ocr_results": ocr_results,
                "temperature_type": temperature_type
            }
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            if callback:
                callback(task_id, "failed", 0, f"Processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def analyze_full_deployment(self, video_path: str, video_name: str, screenshot_dir: Path, temperature_type: str = "room"):
        """Analyze full deployment phase with temperature-specific frame range"""
        try:
            cap_analysis = cv2.VideoCapture(str(video_path))
            mask_area_data = []
            
            # Get temperature-specific frame range
            START_FRAME, END_FRAME = self.get_temperature_frame_range(temperature_type)
            CONFIDENCE_THRESHOLD = 0.5
            
            logger.info(f"Analyzing {temperature_type} temperature: frames {START_FRAME}-{END_FRAME}")
            
            # เลื่อนไปยังเฟรมเริ่มต้น
            cap_analysis.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME - 1)
            
            while cap_analysis.isOpened():
                ret, frame = cap_analysis.read()
                if not ret:
                    break
                    
                frame_count = int(cap_analysis.get(cv2.CAP_PROP_POS_FRAMES))
                
                if frame_count > END_FRAME:
                    break
                
                logger.info(f"Analyzing frame {frame_count} for {temperature_type} temperature full deployment...")
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yolo_results = self.yolo_object_model.predict(
                    source=frame, conf=CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
                )
                
                if yolo_results[0].boxes is not None:
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
                    
                    if len(boxes) > 0:
                        box = boxes[0]
                        x1, y1, x2, y2 = box
                        self.predictor.set_image(image_rgb)
                        masks, scores, _ = self.predictor.predict(
                            box=np.array([x1, y1, x2, y2])[None, :], multimask_output=True
                        )
                        best_mask = masks[np.argmax(scores)]
                        area = np.sum(best_mask)
                        mask_area_data.append((frame_count, area))
            
            cap_analysis.release()
            
            # Find plateau (full deployment)
            if len(mask_area_data) > 0:
                areas = np.array([area for _, area in mask_area_data])
                frames = [frame_num for frame_num, _ in mask_area_data]
                
                if len(areas) >= 3:
                    smoothed_areas = uniform_filter1d(areas, size=3)
                else:
                    smoothed_areas = areas
                
                peak_index = np.argmax(smoothed_areas)
                plateau_frame = frames[peak_index]
                
                threshold = smoothed_areas[peak_index] * 0.99 if smoothed_areas[peak_index] > 0 else 0
                
                for i in range(peak_index, len(smoothed_areas)):
                    if smoothed_areas[i] < threshold:
                        break
                    plateau_frame = frames[i]
                
                logger.info(f"Full Deployment Detected at Frame: {plateau_frame}")
                
                # Capture screenshot
                cap_final = cv2.VideoCapture(str(video_path))
                cap_final.set(cv2.CAP_PROP_POS_FRAMES, plateau_frame - 1)
                ret_final, frame_final = cap_final.read()
                if ret_final:
                    screenshot_path = screenshot_dir / f"Airbag_Full_Deployment_{temperature_type}_frame{plateau_frame}_{video_name}.png"
                    cv2.imwrite(str(screenshot_path), frame_final)
                    logger.info(f"Captured {temperature_type} temperature full deployment screenshot: {screenshot_path}")
                cap_final.release()
                
                return plateau_frame
            
            return None
            
        except Exception as e:
            logger.error(f"Full deployment analysis failed for {temperature_type}: {e}")
            return None

# ===== FASTAPI APP =====
app = FastAPI(title="Airbag Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

# Initialize database and detector
init_db()
detector = AirbagDetector()

# In-memory task tracking
processing_tasks = {}

# ===== HELPER FUNCTIONS =====
def update_task_status(task_id: str, status: str, progress: int, message: str, 
                      output_video_path: str = None, screenshots: List[str] = None, 
                      ocr_results: Dict[str, str] = None):
    """Update task status in database"""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        if status == "completed":
            cursor.execute('''
                UPDATE videos 
                SET status = ?, progress = ?, message = ?, completed_at = ?, 
                    output_video_path = ?, screenshots = ?, ocr_results = ?
                WHERE task_id = ?
            ''', (status, progress, message, datetime.now().isoformat(), 
                  output_video_path, json.dumps(screenshots or []), 
                  json.dumps(ocr_results or {}), task_id))
        else:
            cursor.execute('''
                UPDATE videos 
                SET status = ?, progress = ?, message = ?
                WHERE task_id = ?
            ''', (status, progress, message, task_id))
        
        conn.commit()
        
        # Update in-memory tracking
        processing_tasks[task_id] = {
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to update task status: {e}")
    finally:
        conn.close()

async def process_video_background(video_path: str, task_id: str, temperature_type: str = "room"):
    """Background task for video processing with temperature selection"""
    try:
        result = detector.process_video(video_path, task_id, temperature_type, update_task_status)
        if result["success"]:
            logger.info(f"Video processing completed for task {task_id} with {temperature_type} temperature")
        else:
            logger.error(f"Video processing failed for task {task_id}: {result.get('error')}")
    except Exception as e:
        logger.error(f"Background processing error: {e}")
        update_task_status(task_id, "failed", 0, f"Processing error: {str(e)}")

# ===== API ENDPOINTS =====

@app.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    temperature_type: TemperatureType = TemperatureType.ROOM
):
    """Upload video for processing with temperature selection"""
    
    # Validate file
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only video files are allowed.")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded file with original name
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Save to database with temperature type
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO videos (task_id, original_filename, video_filename, status, message, temperature_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (task_id, file.filename, file.filename, "pending", 
              f"Video uploaded for {temperature_type} temperature analysis, queued for processing", 
              temperature_type))
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()
    
    # Start background processing with temperature type
    background_tasks.add_task(process_video_background, str(file_path), task_id, temperature_type)
    
    return VideoUploadResponse(
        task_id=task_id,
        message=f"Video uploaded successfully for {temperature_type} temperature analysis. Processing started.",
        video_filename=file.filename,
        temperature_type=temperature_type
    )

@app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_processing_status(task_id: str):
    """Get processing status of a task"""
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM videos WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        
        temp_type = 'room'
        if 'temperature_type' in row.keys(): # ตรวจสอบ key ก่อน
            temp_type = row['temperature_type']
        
        return ProcessingStatus(
            task_id=row['task_id'],
            status=row['status'],
            progress=row['progress'],
            message=row['message'],
            created_at=datetime.fromisoformat(row['created_at']),
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            temperature_type=temp_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

@app.get("/videos", response_model=List[VideoRecord])
async def get_video_history(limit: int = Query(default=50, ge=1, le=100)):
    """Get video processing history"""
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT * FROM videos 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        
        records = []
        for row in rows:
            screenshots = json.loads(row['screenshots'] or '[]')
            temp_type = 'room'
            if 'temperature_type' in row.keys(): 
                temp_type = row['temperature_type']
            
            records.append(VideoRecord(
                id=row['id'],
                task_id=row['task_id'],
                original_filename=row['original_filename'],
                video_filename=row['video_filename'],
                status=row['status'],
                progress=row['progress'],
                message=row['message'],
                created_at=datetime.fromisoformat(row['created_at']),
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                output_video_path=row['output_video_path'],
                screenshots=screenshots,
                temperature_type=temp_type
            ))
        
        return records
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

@app.get("/video/{task_id}")
async def get_output_video(task_id: str):
    """Download processed video"""
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT output_video_path FROM videos WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        if not row or not row['output_video_path']:
            raise HTTPException(status_code=404, detail="Output video not found")
        
        video_path = Path(row['output_video_path'])
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found on disk")
        
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=video_path.name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        conn.close()
        
@app.get("/temperature-options")
async def get_temperature_options():
    """Get available temperature options"""
    return {
        "options": [
            {"value": "room", "label": "Room Temperature", "frame_range": "109-121"},
            {"value": "hot", "label": "Hot Temperature", "frame_range": "102-115"},
            {"value": "cold", "label": "Cold Temperature", "frame_range": "112-131"}
        ]
    }

@app.get("/screenshots/{task_id}")
async def get_screenshots(task_id: str):
    """Get list of screenshots for a task"""
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT screenshots FROM videos WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        
        screenshots = json.loads(row['screenshots'] or '[]')
        
        # Convert to relative paths for API response
        screenshot_info = []
        for screenshot_path in screenshots:
            path = Path(screenshot_path)
            if path.exists():
                screenshot_info.append({
                    "filename": path.name,
                    "path": f"/screenshot/{task_id}/{path.name}",
                    "full_path": str(path)
                })
        
        return {"screenshots": screenshot_info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        conn.close()

@app.get("/screenshot/{task_id}/{filename}")
async def get_screenshot(task_id: str, filename: str):
    """Download specific screenshot"""
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT video_filename FROM videos WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        
        video_name = Path(row['video_filename']).stem
        screenshot_path = SCREENSHOT_DIR / video_name / filename
        
        if not screenshot_path.exists():
            raise HTTPException(status_code=404, detail="Screenshot not found")
        
        return FileResponse(
            path=str(screenshot_path),
            media_type="image/png",
            filename=filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        conn.close()

@app.get("/results/{task_id}", response_model=DetectionResult)
async def get_detection_results(task_id: str):
    """Get complete detection results for a task"""
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT * FROM videos WHERE task_id = ?
        ''', (task_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if row['status'] != 'completed':
            raise HTTPException(status_code=400, detail="Task not completed yet")
        
        screenshots = json.loads(row['screenshots'] or '[]')
        ocr_results = json.loads(row['ocr_results'] or '{}')
        
        # Parse processing time (mock value for now)
        processing_time = 120.5  # seconds
        
        return DetectionResult(
            task_id=task_id,
            video_filename=row['video_filename'],
            explosion_frame=18,  # Mock value
            full_deployment_frame=115,  # Mock value
            detected_labels=['FR1', 'FR2', 'RE3'],
            screenshots=screenshots,
            processing_time=processing_time,
            ocr_results=ocr_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        conn.close()

@app.delete("/video/{task_id}")
async def delete_video(task_id: str):
    """Delete video and all associated files"""
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM videos WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Delete files
        video_path = UPLOAD_DIR / row['video_filename']
        if video_path.exists():
            video_path.unlink()
        
        if row['output_video_path']:
            output_path = Path(row['output_video_path'])
            if output_path.exists():
                output_path.unlink()
        
        # Delete screenshots
        screenshots = json.loads(row['screenshots'] or '[]')
        for screenshot_path in screenshots:
            path = Path(screenshot_path)
            if path.exists():
                path.unlink()
        
        # Delete screenshot directory if empty
        video_name = Path(row['video_filename']).stem
        screenshot_dir = SCREENSHOT_DIR / video_name
        if screenshot_dir.exists() and not any(screenshot_dir.iterdir()):
            screenshot_dir.rmdir()
        
        # Delete from database
        cursor.execute('DELETE FROM videos WHERE task_id = ?', (task_id,))
        cursor.execute('DELETE FROM processing_logs WHERE task_id = ?', (task_id,))
        conn.commit()
        
        return {"message": "Video and associated files deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        conn.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "yolo_object": detector.yolo_object_model is not None,
            "yolo_name": detector.yolo_name_model is not None,
            "sam": detector.sam is not None,
            "ocr": detector.ocr.model is not None
        }
    }

# ===== STARTUP EVENT =====
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Airbag Detection API...")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Database: {DB_PATH}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)