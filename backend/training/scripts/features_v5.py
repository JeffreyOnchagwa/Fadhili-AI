"""
Fadhili v5 feature construction — re-export shim.

The implementation lives in `backend/app/services/ksl_features.py` so
that training and the serving API share ONE definition of the input
representation.

This matters more than it looks. The v3 system had the normalization
written twice — once in Python for training, once in TypeScript for the
browser — and the browser faithfully reproduced a normalization bug that
was only ever diagnosed on the Python side. Two implementations of a
preprocessing pipeline will drift, and when they drift the model is
silently fed something it was never trained on.

Training scripts keep importing `features_v5` unchanged; the module now
simply forwards to the canonical one.
"""

import sys
from pathlib import Path

# backend/ must be importable so `app.services` resolves when these
# scripts are run directly from backend/training/scripts.
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.ksl_features import (  # noqa: E402,F401
    EPS,
    POSE_IDS,
    RAW_ROOT,
    FeatureConfig,
    build_background_sequence,
    build_sequence,
    feature_dimension,
    sequence_from_live_frames,
)

__all__ = [
    "EPS",
    "POSE_IDS",
    "RAW_ROOT",
    "FeatureConfig",
    "build_background_sequence",
    "build_sequence",
    "feature_dimension",
    "sequence_from_live_frames",
]
