"""
Fadhili KSL v5 recognition service.

Differences from the v3 service, and why each exists
----------------------------------------------------

1. THE CLIENT SENDS RAW LANDMARKS, NOT FEATURES.
   v3 had the browser normalize landmarks in TypeScript and send the
   finished 150-d feature vectors. That duplicated the preprocessing,
   and the duplicate faithfully reproduced a normalization bug that was
   only ever diagnosed on the Python side. Here the client sends raw
   MediaPipe output and all feature construction happens in
   `ksl_features`, the same module training uses. Verified identical to
   the training path to 0.0 absolute difference.

2. A MOTION GATE ANSWERS "NO SIGN".
   A softmax threshold cannot detect out-of-distribution input: the v3
   model classified pure random noise as "Monday" at 0.71 confidence.
   Measured on real idle footage, a simple pose-motion-energy gate at
   0.010 rejects 99.6% of not-signing windows while wrongly rejecting
   0.0% of real signs. An explicit background CLASS was tried first and
   was worse — it recalled only 46.7% of no-sign windows and cost about
   7 points of classification accuracy — so the gate is used instead.

3. CONFIDENCE IS TEMPERATURE-SCALED.
   Fitted on held-out development signers, never on the diagnostic set.

Honest limitation: the motion gate detects "not moving". It does not
detect "moving but not signing" — a wave, or scratching your head, can
still be classified as a word. That is stated in the API response and
in the UI rather than hidden.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.config import settings
from app.services.ksl_features import FeatureConfig, sequence_from_live_frames

logger = logging.getLogger("fadhili.ksl_recognizer_v5")

# Mean absolute frame-to-frame change across the 24 normalized pose
# channels, below which the window is treated as "nobody is signing".
#
# Chosen from measured data (248 sign windows, 233 idle windows drawn
# from the same recordings): rejects 99.6% of idle, 0.0% of signs.
MOTION_GATE_THRESHOLD = 0.010

# Minimum frames before a prediction is attempted at all.
MIN_FRAMES = 16


@dataclass
class Candidate:
    label: str
    confidence: float


@dataclass
class RecognitionV5Result:
    prediction: Optional[str]
    confidence: float
    accepted: bool
    reason: str
    motion_energy: float
    top_candidates: List[Candidate] = field(default_factory=list)


class KSLRecognizerV5:
    """Loads the v5 champion and serves predictions from raw landmarks."""

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self._model_path = model_path
        self._metadata_path = metadata_path

        self._model = None
        self._classes: List[str] = []
        self._config: Optional[FeatureConfig] = None
        self._temperature: float = 1.0
        self._load_error: Optional[str] = None

        self._load()

    # -----------------------------------------------------------------

    def _load(self) -> None:
        try:
            if not self._metadata_path.is_file():
                raise FileNotFoundError(
                    f"Champion metadata not found: {self._metadata_path}"
                )
            if not self._model_path.is_file():
                raise FileNotFoundError(
                    f"Champion model not found: {self._model_path}"
                )

            with self._metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

            self._classes = list(metadata["classes"])
            self._temperature = float(metadata.get("temperature", 1.0))

            # Rebuild the exact feature configuration the model was
            # trained with. Serving with a different one would be the
            # silent-preprocessing-swap failure all over again.
            stored = metadata["feature_config"]
            known = set(FeatureConfig().describe())
            self._config = FeatureConfig(
                **{k: v for k, v in stored.items() if k in known}
            )

            import keras

            model = keras.models.load_model(str(self._model_path), compile=False)

            expected_dim = int(metadata["feature_dimension"])
            expected_len = int(metadata["sequence_length"])
            shape = model.input_shape

            if len(shape) != 3 or shape[1] != expected_len or shape[2] != expected_dim:
                raise ValueError(
                    f"Model input {shape} disagrees with metadata "
                    f"(None, {expected_len}, {expected_dim})."
                )
            if model.output_shape[-1] != len(self._classes):
                raise ValueError(
                    f"Model outputs {model.output_shape[-1]} classes; "
                    f"metadata lists {len(self._classes)}."
                )

            self._model = model
            self._load_error = None
            logger.info(
                "Fadhili KSL v5 champion loaded: %s classes, input %s, T=%.3f",
                len(self._classes),
                shape,
                self._temperature,
            )

        except Exception as exc:
            self._model = None
            self._load_error = str(exc)
            logger.exception("Failed to load the Fadhili KSL v5 champion.")

    # -----------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    # -----------------------------------------------------------------

    @staticmethod
    def motion_energy(sequence: np.ndarray) -> float:
        """Mean absolute frame-to-frame change of the pose channels."""
        if sequence.shape[0] < 2:
            return 0.0
        return float(np.abs(np.diff(sequence[:, :24], axis=0)).mean())

    def predict(
        self,
        frames: np.ndarray,
        width: float,
        height: float,
    ) -> RecognitionV5Result:
        if self._model is None or self._config is None:
            raise RuntimeError("KSL v5 model is not loaded.")

        frames = np.asarray(frames, dtype=np.float32)

        if frames.shape[0] < MIN_FRAMES:
            return RecognitionV5Result(
                prediction=None,
                confidence=0.0,
                accepted=False,
                reason="insufficient_frames",
                motion_energy=0.0,
            )

        if not np.all(np.isfinite(frames)):
            raise ValueError("Input contains NaN or Infinity.")

        sequence = sequence_from_live_frames(frames, width, height, self._config)

        if sequence is None:
            return RecognitionV5Result(
                prediction=None,
                confidence=0.0,
                accepted=False,
                reason="no_person_detected",
                motion_energy=0.0,
            )

        energy = self.motion_energy(sequence)

        if energy < MOTION_GATE_THRESHOLD:
            return RecognitionV5Result(
                prediction=None,
                confidence=0.0,
                accepted=False,
                reason="no_sign_detected",
                motion_energy=energy,
            )

        probabilities = self._model.predict(
            sequence[np.newaxis, ...], verbose=0
        )[0]

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Model returned NaN or Infinity.")

        probabilities = self._apply_temperature(probabilities)

        order = np.argsort(probabilities)[::-1]
        candidates = [
            Candidate(
                label=self._classes[int(i)],
                confidence=float(probabilities[int(i)]),
            )
            for i in order[:3]
        ]

        best = candidates[0]
        accepted = best.confidence >= settings.KSL_CONFIDENCE_THRESHOLD

        return RecognitionV5Result(
            prediction=best.label if accepted else None,
            confidence=best.confidence,
            accepted=accepted,
            reason="accepted" if accepted else "below_threshold",
            motion_energy=energy,
            top_candidates=candidates,
        )

    def _apply_temperature(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Re-scale a softmax output by the fitted temperature.

        The model's last layer already applied softmax, so recover
        logits with a log before dividing. Any constant offset cancels
        in the softmax that follows, so log(p) is as good as the true
        logits here.
        """
        if abs(self._temperature - 1.0) < 1e-6:
            return probabilities

        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
        scaled = logits / self._temperature
        scaled -= scaled.max()
        exponentiated = np.exp(scaled)
        return exponentiated / exponentiated.sum()


ksl_recognizer_v5 = KSLRecognizerV5(
    model_path=settings.KSL_V5_MODEL_PATH,
    metadata_path=settings.KSL_V5_METADATA_PATH,
)
