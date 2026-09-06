"""API routes for the Fadhili AI backend."""
import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas import (
    CandidatePrediction,
    ClassesResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.ksl_recognizer import DISPLAY_CLASS_NAMES, ksl_recognizer

logger = logging.getLogger("fadhili.routes")

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Reports API and KSL model status. The model is experimental."""
    return HealthResponse(
        status="ok",
        model_loaded=ksl_recognizer.is_ready,
        model_status="experimental",
        supported_languages=["KSL"],
    )


@router.get("/api/v1/ksl/classes", response_model=ClassesResponse, tags=["KSL"])
def get_ksl_classes() -> ClassesResponse:
    """Returns the 30 user-facing KSL classes this model recognizes."""
    return ClassesResponse(language="KSL", classes=DISPLAY_CLASS_NAMES)


@router.post("/api/v1/ksl/predict", response_model=PredictResponse, tags=["KSL"])
def predict_ksl(payload: PredictRequest) -> PredictResponse:
    """
    Runs KSL sign recognition on a 30-frame x 1662-feature sequence.

    This model is experimental and generalizes poorly to new signers —
    a low or even high confidence score is not a guarantee of
    correctness. Predictions below the configured confidence threshold
    are returned as `accepted: false` with `prediction: null`, along
    with the top 3 raw candidates for transparency.
    """
    if not ksl_recognizer.is_ready:
        raise HTTPException(
            status_code=503,
            detail="The KSL recognition model is not currently loaded.",
        )

    try:
        input_array = np.asarray(payload.frames, dtype=np.float32).reshape(
            1, settings.SEQUENCE_LENGTH, settings.FEATURE_LENGTH
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Malformed input frames: {exc}"
        ) from exc

    try:
        result = ksl_recognizer.predict(input_array)
    except Exception:
        logger.exception("KSL inference failed")
        raise HTTPException(status_code=500, detail="Inference failed. Please try again.")

    return PredictResponse(
        language="KSL",
        prediction=result.prediction,
        confidence=result.confidence,
        experimental=True,
        accepted=result.accepted,
        top_candidates=[
            CandidatePrediction(label=c.label, confidence=c.confidence)
            for c in result.top_candidates
        ],
    )