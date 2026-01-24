# DEPLOYMENT.md
## AI-Powered Safety System - Deployment Guide

---

## Quick Start: Local Development

### Prerequisites
```bash
# Install Docker & Docker Compose
brew install docker docker-compose  # macOS
# or download from docker.com

# Python 3.11+
python --version

# Poetry (Python package manager)
pip install poetry
```

### Setup & Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd safety-system

# 2. Copy environment template
cp .env.example .env

# 3. Start infrastructure (PostgreSQL, Redis, Qdrant, etc.)
docker-compose up -d

# 4. Wait for services to be healthy
docker-compose ps
# All services should show "healthy" status

# 5. Initialize database schema
python scripts/bootstrap_db.py

# 6. Seed test data (optional)
python scripts/seed_data.py

# 7. View services
# Event Ingestion:    http://localhost:8001/docs
# Risk Screening:     http://localhost:8002/docs
# Embedding Context:  http://localhost:8003/docs
# Reasoning:          http://localhost:8004/docs
# Action Executor:    http://localhost:8005/docs
# Session Manager:    http://localhost:8006/docs
# Reporting:          http://localhost:8007/docs
#
# Observability:
# Prometheus:         http://localhost:9090
# Grafana:            http://localhost:3000 (admin/dev_password_unsafe_only)
# Jaeger:             http://localhost:16686
```

### Test the System

```bash
# Send a test event to Event Ingestion
curl -X POST http://localhost:8001/v1/events/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key" \
  -d '{
    "event_type": "message_created",
    "platform": "discord",
    "channel_id": "ch123",
    "user_id": "user_target",
    "author_id": "user_harasser",
    "content": "You are a terrible person",
    "metadata": {
      "timestamp": "2026-01-22T10:30:45Z",
      "user_reputation": 0.5,
      "author_reputation": 0.1,
      "channel_history_count": 100
    }
  }'

# Response should be 202 Accepted:
# {
#   "event_id": "evt_...",
#   "status": "queued",
#   "processing_url": "/v1/events/evt_.../status"
# }
```

### Development Workflow

```bash
# Install Python dependencies for local development
cd services/event-ingestion
poetry install

# Run tests
poetry run pytest tests/

# Run linting
poetry run black src/
poetry run isort src/
poetry run mypy src/

# Run a single service (without Docker)
poetry run python -m event_ingestion.main

# View logs
docker-compose logs -f event-ingestion
docker-compose logs -f risk-screening

# Stop all services
docker-compose down

# Clean up volumes (WARNING: deletes all data)
docker-compose down -v
```

---

## Production Deployment: Kubernetes

### Prerequisites

```bash
# Install kubectl
brew install kubectl

# Install Helm
brew install helm

# Install kustomize (alternative to Helm)
brew install kustomize

# Access to Kubernetes cluster
# - GKE: gcloud container clusters create safety-system ...
# - EKS: eksctl create cluster --name safety-system ...
# - AKS: az aks create --resource-group ... --name safety-system ...
```

### Build & Push Docker Images

```bash
# 1. Set registry (e.g., Docker Hub, ECR, GCR)
export REGISTRY=your-registry.azurecr.io
export VERSION=1.0.0

# 2. Build images
docker build -t $REGISTRY/safety-event-ingestion:$VERSION services/event-ingestion/
docker build -t $REGISTRY/safety-risk-screening:$VERSION services/risk-screening/
docker build -t $REGISTRY/safety-embedding-context:$VERSION services/embedding-context/
docker build -t $REGISTRY/safety-reasoning:$VERSION services/reasoning/
docker build -t $REGISTRY/safety-action-executor:$VERSION services/action-executor/
docker build -t $REGISTRY/safety-session-manager:$VERSION services/session-manager/
docker build -t $REGISTRY/safety-reporting:$VERSION services/reporting/

# 3. Push to registry
docker push $REGISTRY/safety-event-ingestion:$VERSION
docker push $REGISTRY/safety-risk-screening:$VERSION
# ... push all images
```

### Deploy with Helm

```bash
# 1. Create namespace
kubectl create namespace safety-system

# 2. Set context
kubectl config set-context --current --namespace=safety-system

# 3. Deploy infrastructure (PostgreSQL, Redis, Qdrant)
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add qdrant https://qdrant.github.io/helm

helm install postgres bitnami/postgresql \
  --set auth.password=prod_password \
  --set primary.persistence.size=100Gi

helm install redis bitnami/redis \
  --set auth.password=prod_password \
  --set master.persistence.size=50Gi

helm install qdrant qdrant/qdrant \
  --set replicaCount=3

# 4. Deploy safety system services
helm install safety-system ./helm/safety-system \
  --values ./helm/safety-system/values-prod.yaml \
  --set image.tag=$VERSION \
  --set image.registry=$REGISTRY

# 5. Wait for deployment
kubectl rollout status deployment/event-ingestion -n safety-system
kubectl rollout status deployment/risk-screening -n safety-system
# ... wait for all services
```

### Deploy with Kustomize

```bash
# 1. Set image versions in kustomization.yaml
cd k8s
kustomize edit set image \
  event-ingestion=$REGISTRY/safety-event-ingestion:$VERSION

# 2. Apply manifests
kustomize build . | kubectl apply -f -

# 3. Monitor rollout
kubectl rollout status deployment -n safety-system
```

### Post-Deployment

```bash
# 1. Verify all pods are running
kubectl get pods -n safety-system

# 2. Check service endpoints
kubectl get services -n safety-system

# 3. Port-forward for testing
kubectl port-forward svc/event-ingestion 8001:8000 -n safety-system

# 4. Verify health
curl http://localhost:8001/health

# 5. Check logs
kubectl logs deployment/event-ingestion -n safety-system -f
kubectl logs deployment/risk-screening -n safety-system -f

# 6. Access Prometheus, Grafana, Jaeger
kubectl port-forward svc/prometheus 9090:9090 -n safety-system
kubectl port-forward svc/grafana 3000:3000 -n safety-system
kubectl port-forward svc/jaeger 16686:16686 -n safety-system
```

---

## Scaling Configuration

### Horizontal Pod Autoscaling (HPA)

Each service includes HPA configuration:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: risk-screening-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: risk-screening
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Auto-scaling behavior:**
- **Screening Service**: Scales up when CPU > 70% or memory > 80%
- **Reasoning Service**: Scales up more aggressively (higher thresholds) due to slower processing
- **Event Ingestion**: Always maintains 3+ replicas (critical path)

### Load Testing

```bash
# Install Locust
pip install locust

# Create locustfile.py (see tests/load_test.py)
# Run load test
locust -f tests/load_test.py \
  --host http://event-ingestion:8000 \
  --users 1000 \
  --spawn-rate 100 \
  --run-time 10m

# Metrics to monitor:
# - Response time (p50, p95, p99)
# - Requests/sec (throughput)
# - Failure rate
# - Queue depth (Redis stream consumer lag)
```

---

## Monitoring & Observability

### Key Metrics to Monitor

```
# Latency Percentiles
safety_screening_duration_ms{quantile="0.99"}  → Should be < 100ms
safety_reasoning_duration_ms{quantile="0.95"}  → Should be < 5000ms

# Throughput
safety_events_processed_total[5m]  → Events/sec

# Queue Health
redis_stream_pending{stream="events:raw"}
redis_stream_pending{stream="events:reasoning_queue"}

# Error Rates
rate(safety_action_failed_total[5m])
rate(safety_llm_timeout_total[5m])

# Model Confidence
histogram_quantile(0.5, safety_screening_confidence)  → Should be > 0.8
```

### Alerting Rules

```yaml
groups:
- name: safety-system-alerts
  rules:
  - alert: HighScreeningLatency
    expr: histogram_quantile(0.99, safety_screening_duration_ms) > 200
    for: 5m
    annotations:
      summary: "Risk screening latency exceeding SLO"

  - alert: QueueBacklog
    expr: redis_stream_pending > 10000
    for: 2m
    annotations:
      summary: "Event queue backlog detected"

  - alert: LLMProviderDown
    expr: rate(safety_llm_timeout_total[5m]) > 0.5
    annotations:
      summary: "LLM provider experiencing errors"

  - alert: HighFalsePositiveRate
    expr: safety_false_positive_rate > 0.1
    for: 1h
    annotations:
      summary: "False positive rate exceeding threshold"
```

---

## Database Migrations

### Create a New Migration

```bash
# Create migration file
alembic revision --autogenerate -m "add user violations table"

# Review migration in alembic/versions/
# Modify if needed (auto-generate is just a helper)

# Apply migration
alembic upgrade head

# Rollback (if needed)
alembic downgrade -1
```

### Backup & Recovery

```bash
# Backup PostgreSQL
pg_dump safety_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore from backup
psql safety_db < backup_20260122_103045.sql

# Backup Qdrant vectors
# (Qdrant provides HTTP backup API)
curl -X POST http://qdrant:6333/collections/harassment_events/snapshots

# See Qdrant docs for full backup/restore procedures
```

---

## Security Checklist

```
☐ API keys rotated (see shared/security/auth.py)
☐ JWT secret changed from default
☐ Database passwords changed from default
☐ Redis auth enabled with strong password
☐ RBAC policies applied (kubectl auth can-i)
☐ Network policies restrict service-to-service communication
☐ TLS/SSL enabled on all external endpoints
☐ HTTPS enforced via ingress controller
☐ Secret management via Vault or cloud provider
☐ Pod security policies enabled
☐ Container image scanning for vulnerabilities (Trivy)
☐ Rate limiting enforced at API gateway
☐ Audit logging enabled (PostgreSQL pgaudit)
☐ Data encryption at rest
☐ Regular security updates scheduled
```

---

## Troubleshooting

### Event Not Processing

```bash
# 1. Check event in Redis stream
redis-cli XRANGE events:raw - +

# 2. Check screening consumer group lag
redis-cli XINFO GROUPS events:raw

# 3. Check if screening service is running
kubectl get pods -l app=risk-screening

# 4. Check logs
kubectl logs deployment/risk-screening

# 5. Verify Redis connectivity
kubectl exec deployment/risk-screening -- \
  redis-cli -h redis ping
```

### High Latency

```bash
# 1. Check service CPU/memory
kubectl top pods -l app=risk-screening

# 2. Check HPA status
kubectl describe hpa risk-screening-hpa

# 3. Check Redis connections
redis-cli info clients

# 4. Check database connections
psql -c "SELECT count(*) FROM pg_stat_activity;"

# 5. Check vector DB query time
# (Monitor Qdrant slow query logs)
```

### Service Not Starting

```bash
# 1. Check pod events
kubectl describe pod <pod-name>

# 2. Check pod logs
kubectl logs <pod-name> --previous  # If crashed

# 3. Check resource requests/limits
kubectl describe deployment <service-name>

# 4. Check probes (liveness, readiness)
kubectl get pod <pod-name> -o yaml | grep -A 10 Probe

# 5. Check environment variables
kubectl get deployment <service-name> -o yaml | grep -A 20 env
```

---

## Performance Tuning

### Connection Pooling

```python
# services/shared/safety_system/db/connection.py
# Adjust pool size based on concurrency
pool_size = 20  # Default connections
max_overflow = 10  # Additional connections under load

# Monitor with:
# SELECT count(*) FROM pg_stat_activity;
```

### Redis Optimization

```bash
# Increase max connections
redis-cli CONFIG SET maxclients 10000

# Monitor memory usage
redis-cli INFO memory

# Enable AOF (append-only file) for durability
redis-cli CONFIG SET appendonly yes
```

### Model Caching

```python
# Risk screening model cached in-memory
# Replace with fresh model every 24 hours

# Monitor model inference time
# (Should be < 50ms for cached model)
```

---

## Maintenance Tasks

### Daily
- Monitor queue depth (alert if > 1000)
- Check false positive rate

### Weekly
- Review audit logs for anomalies
- Update threat intelligence (similar event patterns)

### Monthly
- Performance review (latencies, throughput)
- Security audit
- Database maintenance (ANALYZE, VACUUM)

### Quarterly
- Disaster recovery drill
- Model retraining with new data
- Architecture review

---

## Next Steps

1. **Customize for Your Platform**: Update connectors for Discord, Twitch, etc.
2. **Fine-tune ML Models**: Collect labeled data, retrain screening model
3. **Configure LLM**: Set up OpenAI API key or deploy local LLM
4. **Set Action Policies**: Customize harm-mitigation rules for your platform
5. **Integration Testing**: Run full end-to-end tests
6. **Load Testing**: Verify SLOs under production load
7. **Go Live**: Deploy to staging, then production
