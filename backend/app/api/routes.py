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
    PredictV2Request,
    PredictV2Response,
)
from app.services.ksl_recognizer import DISPLAY_CLASS_NAMES, ksl_recognizer
from app.services.ksl_recognizer_v5 import ksl_recognizer_v5

logger = logging.getLogger("fadhili.routes")

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Reports API and KSL model status. The model is experimental."""
    return HealthResponse(
        status="ok",
        # True when SOMETHING can serve predictions. The v5 champion is
        # preferred; v3 remains the rollback.
        model_loaded=ksl_recognizer_v5.is_ready or ksl_recognizer.is_ready,
        model_status="experimental",
        model_version="v5" if ksl_recognizer_v5.is_ready else "v3",
        supported_languages=["KSL"],
    )


@router.get("/api/v1/ksl/classes", response_model=ClassesResponse, tags=["KSL"])
def get_ksl_classes() -> ClassesResponse:
    """
    Returns the KSL vocabulary the live model recognizes.

    Served from the v5 champion when it is loaded, so the UI can never
    advertise recognition support the deployed model does not have.
    Falls back to the v3 vocabulary otherwise.
    """
    if ksl_recognizer_v5.is_ready:
        return ClassesResponse(language="KSL", classes=ksl_recognizer_v5.classes)
    return ClassesResponse(language="KSL", classes=DISPLAY_CLASS_NAMES)


@router.post(
    "/api/v2/ksl/predict", response_model=PredictV2Response, tags=["KSL"]
)
def predict_ksl_v2(payload: PredictV2Request) -> PredictV2Response:
    """
    Recognizes a KSL sign from RAW browser landmarks (v5 champion).

    The client sends unnormalized MediaPipe output and the server builds
    features with the same code training uses, so preprocessing cannot
    drift between the two.

    A window whose pose motion energy falls below the gate returns
    `accepted: false` with `reason: "no_sign_detected"` rather than
    forcing the input into a vocabulary word. Note the gate detects
    "not moving", not "moving without signing" — an unrelated gesture
    can still be classified.

    Confidence is temperature-scaled but remains uncalibrated enough
    that it must not be shown as a probability.
    """
    if not ksl_recognizer_v5.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "The v5 KSL model is not loaded. "
                f"{ksl_recognizer_v5.load_error or ''}".strip()
            ),
        )

    try:
        result = ksl_recognizer_v5.predict(
            np.asarray(payload.frames, dtype=np.float32),
            payload.width,
            payload.height,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("KSL v5 inference failed")
        raise HTTPException(
            status_code=500, detail="Inference failed. Please try again."
        )

    return PredictV2Response(
        language="KSL",
        model_version="v5",
        prediction=result.prediction,
        confidence=result.confidence,
        experimental=True,
        accepted=result.accepted,
        reason=result.reason,
        motion_energy=result.motion_energy,
        top_candidates=[
            CandidatePrediction(label=c.label, confidence=c.confidence)
            for c in result.top_candidates
        ],
    )


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