# Fadhili AI backend.
#
# Built for a CPU-only host. TensorFlow is the heavy dependency here
# (~500 MB installed), so the image is deliberately kept to the runtime
# essentials: MediaPipe and OpenCV are NOT installed, because landmark
# extraction happens in the user's browser and the server only ever sees
# numeric arrays.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    # TensorFlow spawns one thread per core by default and thrashes on
    # small shared instances. Inference here is a single tiny sequence.
    OMP_NUM_THREADS=2

WORKDIR /app

# Install dependencies first so layer caching survives source changes.
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app ./app
COPY training/models ./training/models

# Non-root user.
RUN useradd --create-home --uid 1000 fadhili \
    && chown -R fadhili:fadhili /app
USER fadhili

# Hosts that inject $PORT (Hugging Face Spaces, Render, Railway) are
# honoured; otherwise default to 8000.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
