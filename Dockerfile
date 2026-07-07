# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 · Build the React/Vite frontend into static assets
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS frontend

WORKDIR /app/frontend

# Install deps against the committed lockfile for reproducible builds.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build the production bundle → /app/frontend/dist
COPY frontend/ ./
RUN npm run build


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 · Python runtime — serves the API *and* the built frontend
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FRONTEND_DIST=/app/frontend/dist \
    HOME=/home/appuser

# System libs required by paddlepaddle (libgomp1) and OpenCV (libglib2.0-0,
# libgl1). PaddleOCR pulls in a full opencv-python build transitively, which
# needs libGL.so.1 (libgl1) despite our pinned opencv-contrib-python-headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code + core CV pipeline.
COPY api/ ./api/
COPY pipeline/ ./pipeline/

# Static frontend bundle from stage 1.
COPY --from=frontend /app/frontend/dist ./frontend/dist

# Run as a non-root user; give it a home for the PaddleOCR model cache.
# Pre-create ~/.paddleocr so a named volume mounted there inherits appuser
# ownership (Docker creates a missing mountpoint as root, which would make the
# startup model download fail under this non-root user).
RUN useradd --create-home --home-dir /home/appuser appuser \
    && mkdir -p /home/appuser/.paddleocr \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8000

# IMPORTANT: single worker only — session state lives in-process (see api/main.py).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
