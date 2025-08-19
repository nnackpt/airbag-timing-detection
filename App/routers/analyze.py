from fastapi import APIRouter
from App.schemas import AnalyzeParams, PipelineResult, FullDeploymentResult, ExplosionResult, LabeledCenters
from App.config import settings
from App.services.pipeline import run_full_pipeline, analyze_full_deployment, detect_centers_and_labels, detect_explosion_17_18

router = APIRouter(prefix="/analyze", tags=["analyze"])

@router.post("/full", response_model=PipelineResult)
def analyze_full(payload: AnalyzeParams):
    params = {
        k: v for k, v in dict(
            confidence_threshold=payload.confidence_threshold,
            motion_threshold=payload.motion_threshold,
            smooth_size=payload.smooth_size,
            plateau_alpha=payload.plateau_alpha,
        ).items() if v is not None
    }
    
    window = None
    if payload.window is not None:
        window = (payload.window.start, payload.window.end)
        
    result = run_full_pipeline(
        payload.video_path,
        condition=payload.condition,
        window=window,
        save_video=payload.save_video,
        params=params
    )
    return result

@router.post("/full-deployment", response_model=FullDeploymentResult)
def only_full_deployment(payload: AnalyzeParams):
    window = None
    if payload.window is not None:
        window = (payload.window.start, payload.window.end)
    elif payload.condition is not None:
        window = settings.CONDITION_WINDOWS[payload.condition]
    else :
        window = settings.CONDITION_WINDOWS["room"]
        
    pf, peak, shot = analyze_full_deployment(
        payload.video_path, window,
        confidence_threshold=payload.confidence_threshold,
        smooth_size=payload.smooth_size,
        plateau_alpha=payload.plateau_alpha
    )
    return { "plateau_frame": pf, "smoothed_peak": peak, "screenshot_path": shot }

@router.post("/centers", response_model=LabeledCenters | dict)
def only_centers(payload: AnalyzeParams):
    labels, _ = detect_centers_and_labels(payload.video_path)
    return labels or {}

@router.post("/explosion", response_model=ExplosionResult)
def only_explosion(payload: AnalyzeParams):
    frame, path = detect_explosion_17_18(payload.video_path, payload.motion_threshold)
    return { "explosion_frame": frame, "screenshot_path": path }