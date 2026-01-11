# Multi-stage build for Railway FastAPI
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies in builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY FastAPI/ ./

# Install runtime utilities needed by healthchecks (docker-compose uses curl)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p models data_prep logs

# Sensible defaults (override via Render/compose envs)
ENV HF_MODEL_REPO=patrickmaina/safaricom-hatespeech-detector \
    HF_INFERENCE_BASE=https://router.huggingface.co/hf-inference \
    USE_LIGHTWEIGHT_MODEL=false \
    ENABLE_MODEL_QUANTIZATION=true \
    MODEL_CACHE_SIZE=1 \
    MAX_MEMORY_MB=2048

USER appuser
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
