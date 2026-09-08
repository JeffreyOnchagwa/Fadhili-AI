from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

# config.py lives at <backend>/app/config.py, so parents[1] is always
# the directory that holds both app/ and training/ — whether that
# directory is called "backend" in this monorepo checkout or is /app
# at the root of the deployed container. Computing it via a
# project-root-then-append-"backend" path (as an earlier version did)
# breaks inside Docker: the Dockerfile copies app/ and training/models
# straight into /app with no "backend" subdirectory, so that approach
# would silently point at a path that doesn't exist and the model would
# fail to load in production.
BACKEND_ROOT = Path(__file__).resolve().parents[1]


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

    # -----------------------------------------------------------------
    # KSL V5 CHAMPION
    # -----------------------------------------------------------------
    #
    # The v5 model takes RAW landmarks and builds features server-side
    # using app.services.ksl_features, so the browser and the training
    # pipeline can never drift apart. V3 above is retained as the
    # rollback model and is still served on the legacy endpoint.

    KSL_V5_MODEL_PATH: Path = Path(
        os.getenv(
            "KSL_V5_MODEL_PATH",
            str(
                BACKEND_ROOT
                / "training"
                / "models"
                / "fadhili_ksl_v5_champion.keras"
            ),
        )
    )

    KSL_V5_METADATA_PATH: Path = Path(
        os.getenv(
            "KSL_V5_METADATA_PATH",
            str(
                BACKEND_ROOT
                / "training"
                / "data"
                / "metadata"
                / "fadhili_v5_champion.json"
            ),
        )
    )

    # Applied to temperature-scaled confidence. Calibration reduces but
    # does not eliminate overconfidence, so this threshold is a coarse
    # guard and must not be presented to users as a probability.
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