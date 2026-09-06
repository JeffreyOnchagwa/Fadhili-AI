"""
Application configuration.

Reads settings from environment variables, with safe defaults for
local development. If a `.env` file is present in the `backend`
directory (copy `.env.example` to `.env` and adjust it), it's loaded
automatically.
"""
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()  # no-op if no .env file exists — safe to call either way

# backend/ directory (parent of the app/ package).
BASE_DIR = Path(__file__).resolve().parent.parent


def _parse_cors_origins(raw: str) -> list[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class Settings:
    """Process-wide configuration, read once at import time."""

    # Path to the KSL model file. Defaults to the model already in the
    # repo; override with KSL_MODEL_PATH if you move or swap it.
    KSL_MODEL_PATH: Path = Path(
        os.getenv(
            "KSL_MODEL_PATH",
            str(BASE_DIR / "ksl-model-source" / "best_model.h5"),
        )
    )

    # Minimum confidence required for a prediction to be "accepted".
    # This model is experimental and prone to confident mistakes, so
    # keep this conservative rather than lowering it to get more hits.
    KSL_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("KSL_CONFIDENCE_THRESHOLD", "0.70")
    )

    # Comma-separated list of allowed CORS origins. Add the deployed
    # Vercel frontend URL here later (e.g. via the CORS_ORIGINS env var).
    CORS_ORIGINS: list[str] = _parse_cors_origins(
        os.getenv("CORS_ORIGINS", "http://localhost:5174")
    )

    # Fixed by the current model's architecture — not user-configurable.
    SEQUENCE_LENGTH: int = 30
    FEATURE_LENGTH: int = 1662


settings = Settings()