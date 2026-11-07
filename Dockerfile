# ==============================================================================
# ChronoStore: Production-Grade Dual-Layer Feature Store Dockerfile
# Multi-stage optimized build for Python 3.11+
# ==============================================================================

# Build Stage: Install dependencies and compile wheels
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install dependencies into wheel cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final Runtime Stage: Minimal attack surface & non-root user
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install curl for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python dependencies from builder
COPY --from=builder /install /usr/local

# Create non-root system user and persistent directories
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --home /app appuser && \
    mkdir -p /app/data/offline && \
    chown -R appuser:appgroup /app

# Copy application source code
COPY --chown=appuser:appgroup . /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    CHRONOSTORE_OFFLINE_STORE_DIR=/app/data/offline \
    CHRONOSTORE_PORT=8000 \
    CHRONOSTORE_HOST=0.0.0.0

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--access-log"]

