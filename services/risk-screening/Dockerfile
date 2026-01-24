FROM python:3.11-slim as builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir poetry==1.7.1
COPY services/risk-screening/pyproject.toml services/risk-screening/poetry.lock* ./
RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi --no-root || pip install fastapi uvicorn pydantic redis asyncpg opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY ./shared /app/shared
COPY ./services/risk-screening/src /app/src
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="/app:/app/src:/app/shared:$PYTHONPATH"
HEALTHCHECK --interval=10s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "risk_screening.main:app", "--host", "0.0.0.0", "--port", "8000"]
