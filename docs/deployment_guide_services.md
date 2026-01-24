# DEPLOYMENT_GUIDE_SERVICES.md

# Event Ingestion + Risk Screening Services - Deployment Guide

Complete production deployment guide for the two core services.

**Table of Contents**
1. [Quick Start (5 minutes)](#quick-start)
2. [Local Development (Docker Compose)](#local-development)
3. [Service Architecture](#service-architecture)
4. [Performance Tuning](#performance-tuning)
5. [Production Deployment (Kubernetes)](#production-deployment)
6. [Monitoring & Observability](#monitoring--observability)
7. [Load Testing](#load-testing)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Start the entire system locally:

```bash
# 1. Clone and setup
git clone <repo>
cd safety-system
cp .env.example .env

# 2. Start services
docker-compose -f docker-compose-extended.yml up -d

# 3. Verify services are running
curl http://localhost:8001/health  # Event Ingestion
curl http://localhost:8002/health  # Risk Screening

# 4. View logs
docker-compose -f docker-compose-extended.yml logs -f event-ingestion
docker-compose -f docker-compose-extended.yml logs -f risk-screening

# 5. Access monitoring dashboards
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Jaeger: http://localhost:16686
```

---

## Local Development

### Prerequisites

```bash
# Docker & Docker Compose
docker --version  # >= 24.0
docker-compose --version  # >= 2.0

# Python 3.11+ (for local development)
python3 --version

# For GPU support (optional)
nvidia-docker --version
```

### Directory Structure

```
safety-system/
├── docker-compose-extended.yml      # Full stack
├── services/
│   ├── event-ingestion/
│   │   ├── Dockerfile
│   │   ├── src/
│   │   │   └── event_ingestion/
│   │   │       ├── main.py           # Main service
│   │   │       ├── models.py
│   │   │       └── validators.py
│   │   ├── requirements.txt
│   │   └── tests/
│   │
│   └── risk-screening/
│       ├── Dockerfile
│       ├── src/
│       │   └── risk_screening/
│       │       ├── main.py           # Main service
│       │       ├── embedding.py
│       │       └── heuristics.py
│       ├── requirements.txt
│       └── tests/
│
├── shared/
│   └── safety_system/
│       ├── core/
│       │   └── models.py             # Shared models
│       └── utils/
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       └── datasources/
│
├── testing/
│   ├── locustfile.py                 # Load tests
│   └── integration_tests.py
│
└── k8s/
    ├── event-ingestion.yaml
    ├── risk-screening.yaml
    ├── helm/
    │   └── Chart.yaml
    └── kustomization.yaml
```

### Environment Configuration

Create `.env` file:

```bash
# PostgreSQL
POSTGRES_PASSWORD=your-secure-password
POSTGRES_USER=safety_user
POSTGRES_DB=safety_db

# Redis
REDIS_PASSWORD=your-redis-password

# Qdrant
QDRANT_API_KEY=your-qdrant-key

# Services
ENVIRONMENT=development
LOG_LEVEL=DEBUG
API_KEYS=test-key,dev-key,local-key

# Event Ingestion
MAX_QUEUE_DEPTH=100000
BATCH_SIZE=500
DEDUP_WINDOW_SECONDS=3600

# Risk Screening
SCREENING_MODEL=sentence-transformers/e5-small-v2
SCREENING_THRESHOLD=0.7
MODEL_BATCH_SIZE=32

# Monitoring
GRAFANA_PASSWORD=your-grafana-password

# Optional: GPU Support
CUDA_VISIBLE_DEVICES=0
```

### Running Locally

```bash
# Start everything
docker-compose -f docker-compose-extended.yml up -d

# Verify all services
docker-compose -f docker-compose-extended.yml ps

# Expected output:
# NAME                    STATUS
# postgres                healthy
# redis                   healthy
# qdrant                  healthy
# prometheus              healthy
# grafana                 healthy
# jaeger                  healthy
# event-ingestion         healthy
# risk-screening          healthy
# locust                  running

# Check service logs
docker-compose -f docker-compose-extended.yml logs event-ingestion
docker-compose -f docker-compose-extended.yml logs risk-screening

# Stop everything
docker-compose -f docker-compose-extended.yml down

# Stop and remove volumes
docker-compose -f docker-compose-extended.yml down -v
```

---

## Service Architecture

### Event Ingestion Service (Port 8001)

**Responsibility:**
- Accept events from multiple platforms
- Validate and normalize
- Detect duplicates
- Apply backpressure
- Stream to Redis

**Flow:**
```
Client Request
    ↓
API Validation (API key check)
    ↓
Event Validation (content length, fields, etc.)
    ↓
Deduplication (check Redis cache)
    ↓
Backpressure Check (queue depth)
    ↓
Add to Redis Stream: events:raw
    ↓
Response to Client (event_id, status, latency)
```

**Performance Targets:**
- P99 Latency: < 100ms
- Throughput: 10,000+ events/sec
- Availability: 99.9%

**Key Endpoints:**

```bash
# Single event
curl -X POST http://localhost:8001/v1/events/ingest \
  -H "Content-Type: application/json" \
  -H "api-key: test-key" \
  -d '{
    "event_id": "evt-123",
    "event_type": "message_created",
    "platform": "discord",
    "channel_id": "ch-123",
    "user_id": "user-123",
    "author_id": "auth-123",
    "content": "This is a test message",
    "created_at": "2025-01-22T10:30:00Z",
    "metadata": {
      "user_reputation": 0.8,
      "author_reputation": 0.6
    }
  }'

# Batch ingest
curl -X POST http://localhost:8001/v1/events/ingest-batch \
  -H "Content-Type: application/json" \
  -H "api-key: test-key" \
  -d '{
    "events": [
      {...},
      {...}
    ]
  }'

# Metrics
curl http://localhost:8001/v1/metrics

# Health
curl http://localhost:8001/health
curl http://localhost:8001/ready
```

### Risk Screening Service (Port 8002)

**Responsibility:**
- Consume events from Redis Stream
- Generate embeddings (e5-small-v2)
- Compute heuristic scores
- Search similar past events
- Route to reasoning service or low-risk queue

**Flow:**
```
Redis Stream: events:raw
    ↓
Consume Batch (max 100 events)
    ↓
For Each Event:
  1. Check embedding cache
  2. Generate/retrieve embedding
  3. Compute heuristic score
  4. Combine scores (60% heuristic + 40% embedding)
  5. Classify risk category
  6. Search similar events
    ↓
Route Based on Risk Score:
  - risk > 0.7 → events:reasoning_queue (HIGH)
  - risk <= 0.7 → events:low_risk (LOW)
    ↓
Acknowledge message
```

**Performance Targets:**
- P99 Latency: < 50ms per event
- Throughput: 5,000+ events/sec
- Model Inference: < 30ms per event

**Key Endpoints:**

```bash
# Screen individual event
curl -X POST http://localhost:8002/v1/risk/screen \
  -H "api-key: test-key" \
  -d 'event_id=evt-123&content=toxic message here'

# Metrics
curl http://localhost:8002/v1/metrics

# Health
curl http://localhost:8002/health
curl http://localhost:8002/ready
```

---

## Performance Tuning

### Event Ingestion Optimization

**Configuration Parameters:**

```yaml
# Batch Processing
BATCH_SIZE: 500              # Events per batch
BATCH_TIMEOUT_MS: 100        # Max wait time

# Queue Management
MAX_QUEUE_DEPTH: 100000      # Max queue size
BACKPRESSURE_THRESHOLD: 0.8  # Trigger at 80% capacity

# Deduplication
DEDUP_WINDOW_SECONDS: 3600   # Cache window
MIN_CONTENT_LENGTH: 1        # Bytes
MAX_CONTENT_LENGTH: 10000    # Bytes
```

**Tuning Strategy:**

```bash
# For high throughput (> 10K/sec)
BATCH_SIZE=1000
BATCH_TIMEOUT_MS=50
WORKERS=8

# For low latency
BATCH_SIZE=100
BATCH_TIMEOUT_MS=10
WORKERS=4

# For memory efficiency
BATCH_SIZE=50
MAX_QUEUE_DEPTH=10000
DEDUP_WINDOW_SECONDS=900
```

### Risk Screening Optimization

**Configuration Parameters:**

```yaml
# Model
SCREENING_MODEL: "sentence-transformers/e5-small-v2"  # 33M params
MODEL_BATCH_SIZE: 32           # Batch size for embedding
DEVICE: "cuda"                 # cpu or cuda
USE_FP32: false                # false = use FP16 (faster)

# Batch Processing
BATCH_SIZE: 100                # Events per batch
BATCH_TIMEOUT_MS: 500          # Max wait time

# Caching
EMBEDDING_CACHE_TTL: 604800    # 7 days
```

**Tuning Strategy:**

```bash
# GPU Acceleration
DEVICE=cuda
MODEL_BATCH_SIZE=64
USE_FP32=false

# CPU-only
DEVICE=cpu
MODEL_BATCH_SIZE=16
USE_FP32=false

# Memory-constrained
DEVICE=cpu
MODEL_BATCH_SIZE=8
USE_FP32=true
```

### Redis Optimization

```bash
# Connect from Docker:
docker exec -it redis redis-cli

# Check memory usage
INFO memory

# Monitor commands
MONITOR

# Check stream size
XLEN events:raw
XLEN events:reasoning_queue
XLEN events:low_risk

# Optimize configurations
CONFIG SET maxmemory 2gb
CONFIG SET maxmemory-policy allkeys-lru
```

---

## Production Deployment

### Kubernetes Setup

**Prerequisites:**

```bash
kubectl version --client
helm version
```

**Install Services:**

```bash
# Using Helm
cd k8s/helm
helm install safety-system ./safety-system-chart

# Using Kustomize
cd k8s
kubectl apply -k .

# Verify deployment
kubectl get pods -n safety
kubectl get services -n safety
```

**Kubernetes Manifests:**

```yaml
# k8s/event-ingestion.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: event-ingestion
  namespace: safety
spec:
  replicas: 3
  selector:
    matchLabels:
      app: event-ingestion
  template:
    metadata:
      labels:
        app: event-ingestion
    spec:
      containers:
      - name: event-ingestion
        image: safety-system/event-ingestion:latest
        ports:
        - containerPort: 8001
        env:
        - name: REDIS_HOST
          value: "redis.safety.svc.cluster.local"
        - name: LOG_LEVEL
          value: "INFO"
        - name: WORKERS
          value: "4"
        - name: MAX_QUEUE_DEPTH
          value: "100000"
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: event-ingestion
  namespace: safety
spec:
  type: LoadBalancer
  selector:
    app: event-ingestion
  ports:
  - protocol: TCP
    port: 8001
    targetPort: 8001

---
# k8s/risk-screening.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: risk-screening
  namespace: safety
spec:
  replicas: 3
  selector:
    matchLabels:
      app: risk-screening
  template:
    metadata:
      labels:
        app: risk-screening
    spec:
      containers:
      - name: risk-screening
        image: safety-system/risk-screening:latest
        ports:
        - containerPort: 8002
        env:
        - name: REDIS_HOST
          value: "redis.safety.svc.cluster.local"
        - name: SCREENING_MODEL
          value: "sentence-transformers/e5-small-v2"
        - name: DEVICE
          value: "cuda"
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "4000m"
            memory: "4Gi"
        # GPU support
        # nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: risk-screening
  namespace: safety
spec:
  type: LoadBalancer
  selector:
    app: risk-screening
  ports:
  - protocol: TCP
    port: 8002
    targetPort: 8002

---
# Auto-scaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: event-ingestion-hpa
  namespace: safety
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: event-ingestion
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Production Checklist

- [ ] Services deployed to Kubernetes
- [ ] Persistent volumes configured (PostgreSQL, Redis)
- [ ] Resource limits set appropriately
- [ ] Auto-scaling configured (HPA)
- [ ] Monitoring and alerting enabled
- [ ] Backup strategy implemented
- [ ] Security policies applied (RBAC, NetworkPolicy)
- [ ] Load balanced endpoints created
- [ ] SSL/TLS certificates configured
- [ ] API keys rotated

---

## Monitoring & Observability

### Prometheus Metrics

Key metrics to monitor:

```promql
# Event Ingestion
rate(events_received_total[5m])          # Events/sec
rate(validation_errors_total[5m])        # Validation errors/sec
histogram_quantile(0.99, ingestion_latency_ms)  # P99 latency
events_queue_depth                       # Queue size
rate(events_deduplicated_total[5m])      # Duplicate rate

# Risk Screening
rate(events_screened_total[5m])          # Screening throughput
histogram_quantile(0.99, screening_latency_ms)  # P99 latency
avg(screening_risk_score)                # Average risk score
```

### Alert Rules

```yaml
# prometheus-alerts.yml
groups:
- name: safety-system
  rules:

  # High queue depth
  - alert: HighQueueDepth
    expr: events_queue_depth > 80000
    for: 5m
    annotations:
      summary: "High event queue depth"

  # High error rate
  - alert: HighValidationErrorRate
    expr: rate(validation_errors_total[5m]) > 100
    for: 2m
    annotations:
      summary: "High validation error rate"

  # High P99 latency
  - alert: HighIngestionLatency
    expr: histogram_quantile(0.99, ingestion_latency_ms) > 500
    for: 5m
    annotations:
      summary: "High ingestion latency"

  # Service down
  - alert: EventIngestionDown
    expr: up{job="event-ingestion"} == 0
    for: 1m
    annotations:
      summary: "Event Ingestion Service is down"
```

### Grafana Dashboards

Available dashboards:

1. **Overview Dashboard** - System health, throughput, latency
2. **Event Ingestion Dashboard** - Queue depth, error rate, deduplication
3. **Risk Screening Dashboard** - Risk distribution, cache hit rate, embedding latency
4. **Infrastructure Dashboard** - CPU, memory, disk, network

---

## Load Testing

### Running Load Tests

```bash
# Start Locust
docker-compose -f docker-compose-extended.yml up locust

# Access web UI
http://localhost:8089

# Run tests programmatically
python -m locust -f testing/locustfile.py \
  --headless \
  --users 1000 \
  --spawn-rate 50 \
  --run-time 5m
```

### Load Test Scenarios

```python
# testing/locustfile.py
class UserBehavior(TaskSet):
    @task(3)
    def ingest_single(self):
        self.client.post("/v1/events/ingest", {
            "event_id": str(uuid4()),
            "event_type": "message_created",
            "platform": "discord",
            "content": "Test message"
        })

    @task(1)
    def ingest_batch(self):
        events = [
            {"event_id": str(uuid4()), ...}
            for _ in range(100)
        ]
        self.client.post("/v1/events/ingest-batch", {"events": events})

# Expected results:
# - 10,000+ events/sec throughput
# - P99 latency < 100ms
# - < 0.1% error rate
```

---

## Troubleshooting

### Common Issues

**1. Services won't start**

```bash
# Check logs
docker-compose logs event-ingestion
docker-compose logs risk-screening

# Verify Redis connection
docker exec redis redis-cli ping

# Verify Qdrant
curl http://localhost:6333/health
```

**2. High queue depth**

```bash
# Check screening service throughput
curl http://localhost:8002/v1/metrics

# Scale up risk-screening replicas
kubectl scale deployment risk-screening --replicas=5

# Monitor
docker-compose logs risk-screening
```

**3. Model loading errors**

```bash
# Check available memory
docker stats risk-screening

# Reduce batch size
SCREENING_MODEL_BATCH_SIZE=8

# Use smaller model
SCREENING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**4. Redis memory issues**

```bash
# Check usage
docker exec redis redis-cli INFO memory

# Set memory limit
docker exec redis redis-cli CONFIG SET maxmemory 2gb

# Enable eviction
docker exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

---

## Summary

| Component | Port | Performance | Resources |
|-----------|------|-------------|-----------|
| Event Ingestion | 8001 | 10K+ evt/sec | 1 CPU, 512MB |
| Risk Screening | 8002 | 5K+ evt/sec | 2 CPU, 2GB (GPU: +1GB) |
| Redis | 6379 | - | 1 CPU, 1GB+ |
| PostgreSQL | 5432 | - | 1 CPU, 1GB |
| Qdrant | 6333 | - | 1 CPU, 500MB |

---

## Next Steps

1. Deploy to production Kubernetes cluster
2. Configure auto-scaling policies
3. Set up monitoring alerts
4. Implement backup strategy
5. Plan disaster recovery
6. Conduct load testing
7. Document runbooks
8. Train operations team
