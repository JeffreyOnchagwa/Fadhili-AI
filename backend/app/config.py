from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

# config.py:
# backend/app/config.py
#
# parents[0] = app
# parents[1] = backend
# parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKEND_ROOT = PROJECT_ROOT / "backend"


class Settings:
    """
    Fadhili AI backend configuration.

    Current KSL recognition model:
        Fadhili KSL V3

    Model input:
        30 frames x 150 features
    """

    # -----------------------------------------------------------------
    # KSL MODEL
    # -----------------------------------------------------------------

    KSL_MODEL_PATH: Path = Path(
        os.getenv(
            "KSL_MODEL_PATH",
            str(
                BACKEND_ROOT
                / "training"
                / "models"
                / "fadhili_ksl_v3_candidate.keras"
            ),
        )
    )

    # Because the current model is substantially overconfident on
    # unfamiliar signers, this threshold is only a basic rejection
    # mechanism. It must not be interpreted as calibrated probability.
    KSL_CONFIDENCE_THRESHOLD: float = float(
        os.getenv(
            "KSL_CONFIDENCE_THRESHOLD",
            "0.70",
        )
    )

    # -----------------------------------------------------------------
    # FRONTEND / CORS
    # -----------------------------------------------------------------

    CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5174,http://127.0.0.1:5174",
        ).split(",")
        if origin.strip()
    ]

    # -----------------------------------------------------------------
    # MODEL INPUT
    # -----------------------------------------------------------------

    SEQUENCE_LENGTH: int = 30

    # V3 uses:
    #
    # Pose       = 24
    # Left hand  = 63
    # Right hand = 63
    #
    # Total      = 150
    FEATURE_LENGTH: int = 150


settings = Settings()