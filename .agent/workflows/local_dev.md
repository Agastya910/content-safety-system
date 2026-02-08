---
description: Run ML services locally with Docker infrastructure only
---

# Hybrid Development Workflow

This workflow runs heavy Python services (ML models, embeddings) locally using `.venv`,
while keeping infrastructure (postgres, redis, qdrant) in Docker.

## Prerequisites
- Activate local venv: `.\.venv\Scripts\activate`
- Ensure dependencies are installed: `pip install -r requirements.txt`

## Step 1: Start Infrastructure Only
// turbo
```bash
docker-compose up -d postgres redis qdrant jaeger
```

## Step 2: Verify Infrastructure Health
// turbo
```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Expected: All containers show "Up" and "(healthy)" where applicable.

## Step 3: Run Services Locally (in separate terminals)

### Terminal 1: Event Ingestion
```bash
.\.venv\Scripts\activate
cd services/event-ingestion
uvicorn src.event_ingestion.main:app --host 0.0.0.0 --port 8001 --reload
```

### Terminal 2: Risk Screening
```bash
.\.venv\Scripts\activate
cd services/risk-screening
uvicorn src.risk_screening.main:app --host 0.0.0.0 --port 8002 --reload
```

### Terminal 3: Embedding Context
```bash
.\.venv\Scripts\activate
cd services/embedding-context
uvicorn src.embedding_context.main:app --host 0.0.0.0 --port 8003 --reload
```

## Step 4: Verify Services
// turbo
```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

## Environment Variables (set before running)
The local services need these environment variables to connect to Docker infrastructure:

```powershell
$env:REDIS_HOST = "localhost"
$env:REDIS_PORT = "6379"
$env:REDIS_PASSWORD = "redis_dev_password"
$env:QDRANT_HOST = "localhost"
$env:QDRANT_PORT = "6333"
$env:QDRANT_API_KEY = "qdrant_dev_key_unsafe_only"
$env:DATABASE_URL = "postgresql+asyncpg://safety:dev_password_unsafe_only@localhost:5432/safety_db"
```

## Stopping
1. Stop local services: Ctrl+C in each terminal
2. Stop infrastructure: `docker-compose down`
