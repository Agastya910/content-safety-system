# Dockerfile
# Optimized Multi-stage build using UV for FastAPI services

# ============ Build Stage ============
FROM ghcr.io/astral-sh/uv:python3.12-slim AS builder

# Enable bytecode compilation and set environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for C-extensions (psycopg2, grpcio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
# --no-dev: Exclude development dependencies for the production image
# --frozen: Use the exact versions from uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ============ Runtime Stage ============
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY ./shared /app/shared
COPY ./src /app/src

# Set environment variables to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:/app/src:$PYTHONPATH" \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run service
# Replace {SERVICE_NAME} with actual service module in your compose file
CMD ["python", "-m", "uvicorn", \
    "{SERVICE_NAME}.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000"]
