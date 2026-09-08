"""Pydantic request/response schemas for the Fadhili AI API."""
import math
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class PredictRequest(BaseModel):
    """
    A sequence of exactly `SEQUENCE_LENGTH` frames, each a list of
    exactly `FEATURE_LENGTH` normalized MediaPipe landmark values.

    For the current model that is 30 frames x 150 values, ordered:
    normalized upper-body pose (24), left hand (63), right hand (63).
    Face landmarks are deliberately not used.

    The exact lengths are read from settings so that promoting a model
    with a different input shape cannot silently disagree with the
    validator below.
    """

    frames: List[List[float]] = Field(
        ...,
        description=(
            "Exactly SEQUENCE_LENGTH frames, each of FEATURE_LENGTH floats."
        ),
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