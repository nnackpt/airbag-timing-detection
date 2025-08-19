from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Tuple

Condition = Literal["room", "hot", "cold"]

class FrameWindow(BaseModel):
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    
class AnalyzeParams(BaseModel):
    video_path: str
    condition: Optional[Condition] = None # For Automatic Map
    window: Optional[FrameWindow] = None # For Manual Map
    confidence_threshold: Optional[float] = None
    motion_threshold: Optional[int] = None
    smooth_size: Optional[int] = None
    plateau_alpha: Optional[float] = None
    save_video: bool = True
    
class CircleCenter(BaseModel):
    x: int
    y: int
    
class LabeledCenters(BaseModel):
    FR1: Tuple[int, int]
    FR2: Tuple[int, int]
    RE3: Tuple[int, int]
    
class ExplosionResult(BaseModel):
    explosion_frame: int
    screenshot_path: Optional[str] = None
    
class FullDeploymentResult(BaseModel):
    plateau_frame: int
    smoothed_peak: int
    screenshot_path: Optional[str] = None
    
class PipelineResult(BaseModel):
    labeled_centers: Optional[LabeledCenters] = None
    explosion: Optional[ExplosionResult] = None
    full_deployment: Optional[FullDeploymentResult] = None
    output_video_path: Optional[str] = None
    saved_frames: int = 0