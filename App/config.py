from pydantic_settings import BaseSettings
from pydantic import computed_field
from typing import Dict, Tuple
import os

class Settings(BaseSettings):
    # Model paths
    YOLO_OBJECT_MODEL_PATH: str
    YOLO_NAME_MODEL_PATH: str
    SAM_CHECKPOINT: str
    SAM_TYPE: str = "vit_h"
    
    # IO
    OUTPUT_ROOT: str = "Output"
    SCREENSHOT_ROOT: str = os.path.join("Output", "Screenshots")
    
    # Params
    CONFIDENCE_THRESHOLD: float = 0.5
    MOTION_THRESHOLD: int = 1500
    SMOOTH_SIZE: int = 3
    PLATEAU_ALPHA: float = 0.98
    
    # Default Windows
    WINDOW_ROOM: str = "109-119"
    WINDOW_HOT: str = "100-113"
    WINDOW_COLD: str = "125-138"
    
    @computed_field
    @property
    def CONDITION_WINDOWS(self) -> Dict[str, Tuple[int, int]]:
        def parse(s: str):
            a, b = s.split("-")
            return (int(a), int(b))
        return {
            "room": parse(self.WINDOW_ROOM),
            "hot": parse(self.WINDOW_HOT),
            "cold": parse(self.WINDOW_COLD)
        }
        
settings = Settings() # Loading from .env