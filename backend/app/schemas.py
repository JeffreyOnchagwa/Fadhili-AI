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


class PredictV2Request(BaseModel):
    """
    RAW MediaPipe landmarks captured live in the browser.

    Unlike PredictRequest, nothing here is normalized client-side. The
    server builds features with the same module training uses, so the
    two can never drift apart — which is exactly what went wrong when
    the browser reimplemented normalization for the v3 model.

    Each frame is 150 raw values:
        [0:24]    6 pose landmarks (MediaPipe 11..16) as x, y, z, visibility
        [24:87]   left hand,  21 landmarks as x, y, z
        [87:150]  right hand, 21 landmarks as x, y, z

    An absent hand is all-zero, matching MediaPipe's own encoding.

    Frame count is variable: the client sends whatever it captured over
    its rolling window, and the server trims to the active interval and
    resamples to the model's sequence length.
    """

    frames: List[List[float]] = Field(
        ...,
        description="16-240 frames of 150 raw landmark values each.",
    )
    width: float = Field(..., gt=0, description="Video pixel width.")
    height: float = Field(..., gt=0, description="Video pixel height.")

    @field_validator("frames")
    @classmethod
    def validate_frames(cls, frames: List[List[float]]) -> List[List[float]]:
        if not 16 <= len(frames) <= 240:
            raise ValueError(
                f"Expected between 16 and 240 frames, got {len(frames)}."
            )
        for i, frame in enumerate(frames):
            if len(frame) != 150:
                raise ValueError(
                    f"Frame {i} has {len(frame)} values; expected exactly 150."
                )
            for value in frame:
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


class PredictV2Response(BaseModel):
    language: str = "KSL"
    model_version: str = "v5"
    prediction: Optional[str] = None
    confidence: float
    experimental: bool = True
    accepted: bool
    #: Why the result came out this way — one of accepted,
    #: below_threshold, no_sign_detected, no_person_detected,
    #: insufficient_frames. Lets the UI explain itself rather than
    #: showing a blank panel.
    reason: str
    #: Pose motion energy of the window, exposed for transparency about
    #: how the no-sign decision was reached.
    motion_energy: float
    top_candidates: List[CandidatePrediction] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_status: str
    #: Which model is actually serving: "v5" (champion) or "v3"
    #: (rollback). Lets the client pick the matching endpoint.
    model_version: str = "v3"
    supported_languages: List[str]


class ClassesResponse(BaseModel):
    language: str = "KSL"
    classes: List[str]