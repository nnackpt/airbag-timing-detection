from fastapi import FastAPI
from App.routers import health, analyze
from App.config import settings

app = FastAPI(title="Airbag Timing API", version="1.0.0")

@app.on_event("startup")
def show_settings():
    print(">> YOLO_OBJECT_MODEL_PATH:", settings.YOLO_OBJECT_MODEL_PATH)
    print(">> YOLO_NAME_MODEL_PATH:", settings.YOLO_NAME_MODEL_PATH)
    print(">> SAM_CHECKPOINT:", settings.SAM_CHECKPOINT)
    
app.include_router(health.router)
app.include_router(analyze.router)

# uvicorn App.main:app --reload --port 8000