"""
Fadhili AI KSL recognition service.

Current experimental model:
    Fadhili KSL V3

Input:
    30 frames x 150 normalized MediaPipe features

Output:
    10 Kenyan Sign Language vocabulary classes

IMPORTANT:
The current model is experimental and has limited cross-signer
generalization. Confidence values should not be interpreted as
calibrated probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import keras
import numpy as np

from app.config import settings


logger = logging.getLogger("fadhili.ksl_recognizer")


# ---------------------------------------------------------------------
# MODEL CLASSES
# ---------------------------------------------------------------------
#
# IMPORTANT:
# This order MUST match the exact class order used during V3 training.
# Do not alphabetically reorder these labels unless the training
# pipeline itself used a different order.
#
# The V3 training dataset used these 10 classes.
# ---------------------------------------------------------------------

INTERNAL_CLASS_NAMES: List[str] = [
    "Agreement",
    "Friend",
    "Gift",
    "Market",
    "Monday",
    "Picture",
    "Proud",
    "Teach",
    "Twin",
    "Ugali",
]


DISPLAY_CLASS_NAMES: List[str] = INTERNAL_CLASS_NAMES.copy()


def _to_display(internal_label: str) -> str:
    """
    Convert an internal model class label into the label displayed
    to the user.

    Currently the internal and display labels are identical.
    """
    return internal_label


# ---------------------------------------------------------------------
# RESPONSE OBJECTS
# ---------------------------------------------------------------------


@dataclass
class PredictionCandidate:
    label: str
    confidence: float


@dataclass
class RecognitionResult:
    prediction: Optional[str]
    confidence: float
    accepted: bool
    top_candidates: List[PredictionCandidate] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------
# RECOGNIZER
# ---------------------------------------------------------------------


class KSLRecognizer:
    """
    Loads the Fadhili KSL model once and serves predictions.
    """

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float,
    ) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold

        self._model = None
        self._load_error: Optional[str] = None

        self._load_model()

    # -----------------------------------------------------------------
    # MODEL LOADING
    # -----------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            if not self._model_path.is_file():
                raise FileNotFoundError(
                    f"Model file not found: "
                    f"{self._model_path}"
                )

            logger.info(
                "Loading Fadhili KSL model from %s",
                self._model_path,
            )

            model = keras.models.load_model(
                str(self._model_path),
                compile=False,
            )

            # ---------------------------------------------------------
            # Validate model input
            # ---------------------------------------------------------

            input_shape = model.input_shape

            expected_input_shape = (
                None,
                settings.SEQUENCE_LENGTH,
                settings.FEATURE_LENGTH,
            )

            if len(input_shape) != 3:
                raise ValueError(
                    f"Unexpected model input shape: "
                    f"{input_shape}"
                )

            if (
                input_shape[1]
                != settings.SEQUENCE_LENGTH
                or input_shape[2]
                != settings.FEATURE_LENGTH
            ):
                raise ValueError(
                    f"Model input shape {input_shape} "
                    f"does not match expected "
                    f"{expected_input_shape}."
                )

            # ---------------------------------------------------------
            # Validate model output
            # ---------------------------------------------------------

            output_shape = model.output_shape

            expected_classes = len(
                INTERNAL_CLASS_NAMES
            )

            if len(output_shape) != 2:
                raise ValueError(
                    f"Unexpected model output shape: "
                    f"{output_shape}"
                )

            if output_shape[-1] != expected_classes:
                raise ValueError(
                    f"Model output has "
                    f"{output_shape[-1]} classes; "
                    f"expected {expected_classes}."
                )

            self._model = model
            self._load_error = None

            logger.info(
                "Fadhili KSL model loaded successfully. "
                "input_shape=%s output_shape=%s",
                input_shape,
                output_shape,
            )

        except Exception as exc:
            self._model = None
            self._load_error = str(exc)

            logger.exception(
                "Failed to load Fadhili KSL model."
            )

    # -----------------------------------------------------------------
    # STATUS
    # -----------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # -----------------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------------

    def predict(
        self,
        frames: np.ndarray,
    ) -> RecognitionResult:
        """
        Run KSL inference.

        Expected input shape:

            (1, 30, 150)

        Returns:
            RecognitionResult
        """

        if self._model is None:
            raise RuntimeError(
                "KSL model is not loaded."
            )

        frames = np.asarray(
            frames,
            dtype=np.float32,
        )

        expected_shape = (
            1,
            settings.SEQUENCE_LENGTH,
            settings.FEATURE_LENGTH,
        )

        if frames.shape != expected_shape:
            raise ValueError(
                f"Prediction input has shape "
                f"{frames.shape}; expected "
                f"{expected_shape}."
            )

        if not np.all(np.isfinite(frames)):
            raise ValueError(
                "Prediction input contains "
                "NaN or Infinity."
            )

        # -------------------------------------------------------------
        # MODEL INFERENCE
        # -------------------------------------------------------------

        probabilities = self._model.predict(
            frames,
            verbose=0,
        )[0]

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        if probabilities.shape != (
            len(INTERNAL_CLASS_NAMES),
        ):
            raise ValueError(
                f"Model returned prediction shape "
                f"{probabilities.shape}; expected "
                f"({len(INTERNAL_CLASS_NAMES)},)."
            )

        if not np.all(
            np.isfinite(probabilities)
        ):
            raise ValueError(
                "Model returned NaN or Infinity."
            )

        # -------------------------------------------------------------
        # RANK RESULTS
        # -------------------------------------------------------------

        ranked_indices = np.argsort(
            probabilities
        )[::-1]

        top_candidates = [
            PredictionCandidate(
                label=_to_display(
                    INTERNAL_CLASS_NAMES[
                        int(index)
                    ]
                ),
                confidence=float(
                    probabilities[int(index)]
                ),
            )
            for index in ranked_indices[:3]
        ]

        best_index = int(
            ranked_indices[0]
        )

        best_label = _to_display(
            INTERNAL_CLASS_NAMES[
                best_index
            ]
        )

        best_confidence = float(
            probabilities[best_index]
        )

        # -------------------------------------------------------------
        # CONFIDENCE REJECTION
        # -------------------------------------------------------------

        accepted = (
            best_confidence
            >= self._confidence_threshold
        )

        prediction: Optional[str]

        if accepted:
            prediction = best_label
        else:
            prediction = None

        return RecognitionResult(
            prediction=prediction,
            confidence=best_confidence,
            accepted=accepted,
            top_candidates=top_candidates,
        )


# ---------------------------------------------------------------------
# SINGLETON
# ---------------------------------------------------------------------
#
# Loaded once when FastAPI imports this module.
# ---------------------------------------------------------------------

ksl_recognizer = KSLRecognizer(
    model_path=settings.KSL_MODEL_PATH,
    confidence_threshold=(
        settings.KSL_CONFIDENCE_THRESHOLD
    ),
)