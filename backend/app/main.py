"""
Fadhili AI backend — FastAPI application entry point.

Run from the `backend` directory with:
    uvicorn app.main:app --reload --port 8000
"""
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fadhili.main")

app = FastAPI(
    title="Fadhili AI API",
    description=(
        "Backend for Fadhili AI, an AI-powered sign-language "
        "accessibility platform. Kenyan Sign Language (KSL) recognition "
        "is currently experimental and generalizes poorly to signers "
        "outside its training data. ASL and BSL recognition are not "
        "yet implemented."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A full prediction request is well under a megabyte as JSON, so
# anything far larger is rejected before the body is parsed. This is a
# coarse, best-effort guard (it relies on a Content-Length header, so it
# won't catch chunked-encoded bodies) — the strict shape validation in
# PredictRequest is the real correctness check.
MAX_CONTENT_LENGTH_BYTES = 2_000_000  # 2 MB


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_CONTENT_LENGTH_BYTES:
                return JSONResponse(
                    status_code=413, content={"detail": "Request body too large."}
                )
        except ValueError:
            pass  # malformed header — let normal parsing handle it
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak stack traces or internal error details to clients.
    logger.exception("Unhandled exception while processing %s", request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


app.include_router(router)


@app.get("/", tags=["System"])
def root():
    """Basic Fadhili AI API information."""
    return {
        "name": "Fadhili AI API",
        "description": "Sign-language accessibility API. KSL recognition is experimental.",
        "version": "0.1.0",
        "docs": "/docs",
    }