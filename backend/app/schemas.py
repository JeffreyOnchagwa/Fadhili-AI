"""Pydantic request/response schemas for the Fadhili AI API."""
import math
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class PredictRequest(BaseModel):
    """
    A sequence of exactly `SEQUENCE_LENGTH` (30) frames, each a list of
    exactly `FEATURE_LENGTH` (1662) MediaPipe Holistic features, in the
    fixed order: pose, face, left hand, right hand.
    """

    frames: List[List[float]] = Field(
        ...,
        description="Exactly 30 frames, each a list of exactly 1662 floats.",
    )

    @field_validator("frames")
    @classmethod
    def validate_frames(cls, frames: List[List[float]]) -> List[List[float]]:
        if len(frames) != settings.SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected exactly {settings.SEQUENCE_LENGTH} frames, "
                f"got {len(frames)}."
            )
        for i, frame in enumerate(frames):
            if len(frame) != settings.FEATURE_LENGTH:
                raise ValueError(
                    f"Frame {i} has {len(frame)} values; expected exactly "
                    f"{settings.FEATURE_LENGTH}."
                )
            for value in frame:
                # Pydantic has already coerced each value to a float by
                # the time this runs; explicitly reject NaN/Infinity,
                # which Python's JSON parser accepts by default but
                # which would silently corrupt model input.
                if math.isnan(value) or math.isinf(value):
                    raise ValueError(f"Frame {i} contains NaN or Infinity.")
        return frames


class CandidatePrediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    language: str = "KSL"
    prediction: Optional[str] = None
    confidence: float
    experimental: bool = True
    accepted: bool
    top_candidates: List[CandidatePrediction] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_status: str
    supported_languages: List[str]


class ClassesResponse(BaseModel):
    language: str = "KSL"
    classes: List[str]