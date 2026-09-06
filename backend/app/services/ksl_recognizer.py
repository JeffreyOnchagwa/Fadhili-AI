"""
KSL recognizer service.

Wraps the pretrained Keras KSL classifier so the rest of the API never
touches TensorFlow/Keras directly. The model loads once, when this
module is first imported (i.e. once per server process, not per
request), and exposes a small, stable interface: `is_ready` and
`predict()`.

IMPORTANT — model quality: this pretrained model generalizes poorly to
signers it wasn't trained on and can be confidently wrong. It is
explicitly experimental. Nothing built on top of this service may
present its output as reliable, and predictions below the configured
confidence threshold are never surfaced as accepted results.

Replacing the model later: swap what `_load_model` builds as
`self._model`, and update `INTERNAL_CLASS_NAMES` if the new model's
classes differ. The public interface (`is_ready`, `predict`,
`load_error`) should not need to change.
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

# Exact model class order — fixed by how the model was trained.
# Do not reorder. Includes the source model's original misspellings
# ("cousing", "doughter"), preserved because they are the literal
# trained class indices.
INTERNAL_CLASS_NAMES: List[str] = [
    "me", "you", "friend", "name", "mine", "who", "how", "please",
    "help-me", "wait", "now", "home", "where", "give-me", "thank-you",
    "polite", "hello", "good", "mother", "father", "uncle", "cousing",
    "brother", "sister", "doughter", "parent", "relative", "yes", "no",
    "sorry",
]

# Display-only corrections for the two misspelled internal labels.
# Internal training labels stay untouched; only user-facing output
# is corrected.
_DISPLAY_OVERRIDES = {"cousing": "cousin", "doughter": "daughter"}

DISPLAY_CLASS_NAMES: List[str] = [
    _DISPLAY_OVERRIDES.get(name, name) for name in INTERNAL_CLASS_NAMES
]


def _to_display(internal_label: str) -> str:
    return _DISPLAY_OVERRIDES.get(internal_label, internal_label)


@dataclass
class PredictionCandidate:
    label: str
    confidence: float


@dataclass
class RecognitionResult:
    prediction: Optional[str]
    confidence: float
    accepted: bool
    top_candidates: List[PredictionCandidate] = field(default_factory=list)


class KSLRecognizer:
    """Loads the KSL model once and serves predictions from it."""

    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._load_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            if not self._model_path.is_file():
                raise FileNotFoundError(f"Model file not found: {self._model_path}")

            logger.info("Loading KSL model from %s", self._model_path)
            # Standalone Keras 3 load — deliberately NOT
            # tensorflow.keras.models.load_model, which fails on this
            # model under TF Keras 2.15 (InputLayer batch_shape issue).
            model = keras.models.load_model(str(self._model_path), compile=False)

            expected_classes = len(INTERNAL_CLASS_NAMES)
            output_shape = model.output_shape
            if output_shape[-1] != expected_classes:
                raise ValueError(
                    f"Model output has {output_shape[-1]} classes; "
                    f"expected {expected_classes}."
                )

            input_shape = model.input_shape
            if (
                input_shape[1] != settings.SEQUENCE_LENGTH
                or input_shape[2] != settings.FEATURE_LENGTH
            ):
                raise ValueError(
                    f"Model input shape {input_shape} does not match the "
                    f"expected (None, {settings.SEQUENCE_LENGTH}, "
                    f"{settings.FEATURE_LENGTH})."
                )

            self._model = model
            logger.info(
                "KSL model loaded. input_shape=%s output_shape=%s",
                input_shape,
                output_shape,
            )
        except Exception as exc:
            # Loading must never crash the whole app — /health should
            # still be able to report "model not loaded" cleanly.
            self._load_error = str(exc)
            logger.error("Failed to load KSL model: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def predict(self, frames: np.ndarray) -> RecognitionResult:
        """
        Runs inference on a (1, 30, 1662) float32 array.

        Raises RuntimeError if the model isn't loaded — callers should
        check `is_ready` first and return a clean 503 instead of
        calling this.
        """
        if self._model is None:
            raise RuntimeError("KSL model is not loaded.")

        probabilities = self._model.predict(frames, verbose=0)[0]

        ranked_indices = np.argsort(probabilities)[::-1]
        top_candidates = [
            PredictionCandidate(
                label=_to_display(INTERNAL_CLASS_NAMES[idx]),
                confidence=float(probabilities[idx]),
            )
            for idx in ranked_indices[:3]
        ]

        best_idx = int(ranked_indices[0])
        best_confidence = float(probabilities[best_idx])
        accepted = best_confidence >= self._confidence_threshold

        return RecognitionResult(
            prediction=_to_display(INTERNAL_CLASS_NAMES[best_idx]) if accepted else None,
            confidence=best_confidence,
            accepted=accepted,
            top_candidates=top_candidates,
        )


# Singleton — instantiated once when this module is first imported,
# so the (relatively slow) model load happens once per server process.
ksl_recognizer = KSLRecognizer(
    model_path=settings.KSL_MODEL_PATH,
    confidence_threshold=settings.KSL_CONFIDENCE_THRESHOLD,
)