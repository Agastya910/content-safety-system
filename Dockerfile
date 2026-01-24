# Dockerfile
# Multi-stage build for all FastAPI services

# ============ Build Stage ============
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy pyproject.toml and poetry.lock
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# ============ Runtime Stage ============
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
    /usr/local/lib/python3.11/site-packages

# Copy application code
COPY ./shared /app/shared
COPY ./src /app/src

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/src:$PATH" \
    PYTHONPATH="/app:/app/src:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run service
# Replace {SERVICE_NAME} with actual service module
CMD ["python", "-m", "uvicorn", \
     "{SERVICE_NAME}.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]

# ============ Development Stage (alternative) ============
FROM python:3.11-slim as dev

WORKDIR /app

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy project files
COPY . .

# Install all dependencies (including dev)
RUN poetry config virtualenvs.create false && \
    poetry install --with dev

# Set Python path
ENV PYTHONPATH="/app:/app/src:$PYTHONPATH"

# Run tests by default in dev
CMD ["poetry", "run", "pytest", "-v", "tests/"]
