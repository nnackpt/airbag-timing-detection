import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from App.config import settings

class ModelBundle:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_object = YOLO(settings.YOLO_OBJECT_MODEL_PATH)
        self.yolo_name = YOLO(settings.YOLO_NAME_MODEL_PATH)
        sam = sam_model_registry[settings.SAM_TYPE](checkpoint=settings.SAM_CHECKPOINT)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)
        
_models: ModelBundle | None = None

def get_models() -> ModelBundle:
    global _models
    if _models is None:
        _models = ModelBundle()
    return _models