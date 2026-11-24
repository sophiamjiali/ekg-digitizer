# Serving image for the EKG digitizer API (built by .github/workflows/deploy.yml).
# Local run:
#   docker build -t ekg-digitizer .
#   docker run -p 8080:8080 -e WANDB_API_KEY=... ekg-digitizer
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EKG_CONFIG=/app/configs/config.yaml \
    EKG_MODEL_CACHE_DIR=/tmp/ekg-model-cache \
    WANDB_DIR=/tmp/wandb \
    WANDB_CACHE_DIR=/tmp/wandb-cache \
    WANDB_CONFIG_DIR=/tmp/wandb-config

# libgl1/libglib2.0-0: runtime libraries OpenCV needs on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (the default wheel bundles several GB of CUDA); this layer is
# cached until the dependency list changes
COPY pyproject.toml README.md LICENSE ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY src ./src
COPY configs ./configs
RUN pip install ".[serve]"

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8080
# Cloud Run sets $PORT; default to 8080 elsewhere. One worker: the model is held in
# memory per process, so scale with instances rather than workers.
CMD ["sh", "-c", "exec uvicorn ekg_digitizer.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]