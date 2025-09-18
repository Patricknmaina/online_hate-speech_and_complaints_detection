# Railway-optimized Dockerfile for FastAPI Backend
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python dependencies (with system build tools if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

COPY FastAPI/ .

RUN mkdir -p models data_prep && chmod -R 755 /app
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Health check (make sure you have a /health route in FastAPI)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

EXPOSE 8000

# Corrected entrypoint for FastAPI in subfolder
CMD uvicorn FastAPI.main:app --host 0.0.0.0 --port ${PORT:-8000}
