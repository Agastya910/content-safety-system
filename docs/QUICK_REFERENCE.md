# QUICK_REFERENCE.md
## AI Safety System - Architecture Cheat Sheet

---

## 📊 System Overview (One Diagram)

```
PLATFORMS (Discord, Twitch, Web)
    ↓
┌─────────────────────────────────────────┐
│  EVENT INGESTION SERVICE (8001)         │
│  - Normalize webhooks                   │
│  - Deduplicate                          │
│  - Queue to Redis Streams               │
└──────────────────┬──────────────────────┘
                   ↓
            events:raw stream
                   ↓
        ┌─────────────────────┐
        │ CRITICAL PATH       │
        │ (< 100ms SLO)       │
        └────────┬────────────┘
                 ↓
    ┌────────────────────────────┐
    │ RISK SCREENING (8002)      │
    │ - TinyBERT model           │
    │ - Rule-based checks        │
    │ - Returns risk score       │
    └────────┬───────────────────┘
             ↓
        if risk > 0.7:
        events:reasoning_queue
             ↓
        ┌─────────────────────────────────────────────────────┐
        │ ASYNC PATH (< 5s latency, but not critical)         │
        └─────────────────────────────────────────────────────┘
             ↓ ┌──────────────────────────────────────┐
             │ │ EMBEDDING + CONTEXT (8003)           │
             │ │ - Embed text → OpenAI/ST              │
             │ │ - Similarity search (Qdrant)          │
             │ │ - Retrieve similar incidents          │
             │ │ - Session aggregation                 │
             │ └──────────┬───────────────────────────┘
             │            ↓
             │      (similar_events + session_context)
             │            ↓
             └────→ ┌────────────────────────────────┐
                    │ REASONING (8004)               │
                    │ - LLM (OpenAI or local)        │
                    │ - Chain-of-thought             │
                    │ - Return action recommendation │
                    └────────┬─────────────────────┘
                             ↓
                    events:actions_pending
                             ↓
              ┌──────────────────────────────┐
              │ ACTION EXECUTOR (8005)       │
              │ - Apply action (mute/timeout)│
              │ - Platform API integration   │
              │ - Audit logging              │
              └──────────┬───────────────────┘
                         ↓
              ┌──────────────────────────────┐
              │ POSTGRES + REDIS + QDRANT    │
              │ - Audit trail                │
              │ - User violations            │
              │ - Vector embeddings          │
              │ - Cache + state              │
              └──────────────────────────────┘
```

---

## 🎯 The 7 Services At A Glance

| Service | Port | Purpose | Input | Output | Tech |
|---------|------|---------|-------|--------|------|
| **Event Ingestion** | 8001 | Accept webhooks | HTTP POST | Redis events:raw | FastAPI |
| **Risk Screening** | 8002 | Classify risk (100ms) | Text | Risk score | ML model |
| **Embedding Context** | 8003 | Find similar patterns | Event ID | Similar events | Vector DB |
| **Reasoning** | 8004 | LLM analysis (RAG) | Risk + Context | Action recommendation | LLM |
| **Action Executor** | 8005 | Apply mitigation | Recommended action | Platform changes | SDK calls |
| **Session Manager** | 8006 | Track conversations | Events | Session state | Redis |
| **Reporting** | 8007 | Analytics & metrics | Events | Dashboard data | SQL queries |

---

## 🔄 Event Flow Example

### Scenario: Harassment Detection & Mitigation

```
1. USER A writes: "You're trash at the game"
   → Webhook sent to Event Ingestion Service
   
2. Event Ingestion
   ├─ Normalizes: {user_id, author_id, content, metadata}
   ├─ Deduplicates: Check if event_id seen before
   └─ Publishes: → events:raw stream
   
3. Risk Screening Service (consumer)
   ├─ Reads: event from events:raw
   ├─ ML Model inference: "harassment" score = 0.78
   └─ Publishes: → events:reasoning_queue (risk > 0.7 threshold)
   
4. Embedding Service (consumer)
   ├─ Generates embedding for "You're trash at the game"
   ├─ Searches Qdrant: Find similar past incidents
   ├─ Returns: [
   │    {event_id: evt_2024_jan_10, similarity: 0.91, action: timeout_24h},
   │    {event_id: evt_2024_jan_15, similarity: 0.87, action: warned}
   │  ]
   └─ Aggregates: Session data (12 msgs in 8 min, escalation_score: 0.76)
   
5. Reasoning Service (consumer)
   ├─ Prompts LLM: "Event: '{content}', Similar: [similar_events], History: [user history]"
   ├─ LLM Response: "Pattern matches 91% similar incident that resolved with 24h timeout. User has prior warning. Recommend: timeout_24h"
   ├─ Confidence: 0.88
   └─ Publishes: → events:actions_pending
   
6. Action Executor Service (consumer)
   ├─ Reads: Recommended action = timeout_24h
   ├─ Calls: Discord API → Mute user for 24 hours
   ├─ Logs: Audit trail with evidence traces
   ├─ Publishes: → events:actions_applied
   └─ Stores: Action in PostgreSQL
   
7. Platform User Experience
   └─ User A sees: "You've been timed out for 24 hours"
   
8. Session Manager
   ├─ Updates session: harassment_flags_count += 1
   ├─ Monitors: Escalation trend
   └─ Alerts: If rapid escalation detected
```

---

## 📦 Data Models (Key Fields)

### Event
```python
Event {
  event_id: str              # Unique ID
  platform: "discord"        # Where from
  user_id: str               # Target/recipient
  author_id: str             # Sender
  content: str               # Message text
  metadata: {
    timestamp: datetime
    user_reputation: 0.0-1.0
    author_reputation: 0.0-1.0
  }
}
```

### RiskPrediction
```python
RiskPrediction {
  event_id: str
  risk_score: 0.0-1.0        # ML confidence
  risk_category: "targeted_harassment"
  confidence: 0.0-1.0
  flags: ["repeated_name_calling", "targeting_behavior"]
  screening_time_ms: 45
}
```

### Action
```python
Action {
  action_id: str
  action_type: "timeout"     # warning|timeout|mute|ban
  duration_hours: 24
  user_id: str               # Who to punish
  platform: "discord"
  reason_code: "targeted_harassment"
  reasoning: "94% similar incident..."
  evidence_chain: [...]      # Traces of why
}
```

### Session
```python
Session {
  session_id: str
  participants: [user_id, ...]
  message_count: 12
  time_span_minutes: 8
  escalation_score: 0.76     # Temporal signal
  harassment_flags_count: 3
  temporal_features: {
    intervals: [45s, 30s, 25s],  # Time between messages
    risk_scores: [0.3, 0.5, 0.8] # Increasing!
  }
}
```

---

## 🗄️ Database Schema (Simplified)

### PostgreSQL
```sql
-- Users & Violations
users (id, user_id, platform, reputation, violation_count, ...)

-- Events Log
events (id, event_id, platform, content, risk_score, ...)

-- Sessions
sessions (id, session_id, channel_id, escalation_score, ...)

-- Actions Taken
actions (id, action_id, event_id, action_type, applied_at, ...)

-- Audit Trail
audit_log (id, action_id, actor, details, created_at, ...)
```

### Redis Streams
```
events:raw              # Raw incoming events
events:screened         # After risk screening
events:reasoning_queue  # Pending reasoning
events:actions_pending  # Pending execution
events:actions_applied  # Completed actions
events:dlq_errors       # Dead letter queue
```

### Redis Cache
```
embedding:{event_id} → [0.123, -0.456, ...]  # 7 day TTL
user:reputation:{user_id} → 0.75             # 1 hour TTL
session:{session_id} → {...}                 # 30 min TTL
```

### Qdrant Vector DB
```
Collection: harassment_events
  Vector: 1536-dim (OpenAI embedding)
  Payload:
    - event_id
    - user_id
    - timestamp
    - risk_score
    - action_taken
    - action_outcome
```

---

## 🚀 Deployment Quick Commands

### Local (Docker Compose)
```bash
docker-compose up -d           # Start all services
docker-compose logs -f         # Follow logs
docker-compose ps              # Check status
docker-compose down -v         # Stop + delete volumes

# Test
curl -X POST http://localhost:8001/v1/events/ingest \
  -H "X-API-Key: test" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### Kubernetes (Helm)
```bash
helm install safety-system ./helm/safety-system \
  --namespace safety-system \
  --values ./helm/safety-system/values-prod.yaml

kubectl get pods -n safety-system
kubectl logs deployment/risk-screening -n safety-system -f
kubectl port-forward svc/prometheus 9090:9090 -n safety-system
```

---

## 📊 Key Metrics to Monitor

```
Latency:
  safety_screening_duration_ms{quantile="0.99"} < 100ms
  safety_reasoning_duration_ms{quantile="0.95"} < 5000ms

Throughput:
  rate(safety_events_processed_total[5m]) > 1000/sec

Queue Health:
  redis_stream_pending{stream="events:raw"} < 1000
  redis_stream_consumer_lag{group="reasoning"} < 5 sec

Accuracy:
  safety_false_positive_rate < 0.08
  histogram_quantile(0.5, safety_screening_confidence) > 0.8

Errors:
  rate(safety_action_failed_total[5m]) < 0.01
  rate(safety_llm_timeout_total[5m]) < 0.05
```

---

## 🔧 Configuration Template (.env)

```bash
# Service Env
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_password

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/safety_db

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_API_KEY=qdrant_key

# LLM
LLM_PROVIDER=local              # or "openai"
LOCAL_LLM_ENDPOINT=http://ollama:11434
OPENAI_API_KEY=sk-...

# ML Model
SCREENING_MODEL=tinybert
SCREENING_THRESHOLD=0.7

# Action Policy
ACTION_POLICY=graduated_escalation  # or "severity_based"

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
SENTRY_DSN=https://...

# Auth
JWT_SECRET=super_secret
API_KEYS=key1,key2,key3
```

---

## 🎯 Extension Points (Plugin Architecture)

### 1. Add New Risk Model
```python
# services/risk-screening/src/risk_screening/models/
class CustomBERTModel(RiskClassifier):
    async def predict(self, text, context):
        # Your logic
        return RiskPrediction(...)

# Update config to use it
SCREENING_MODEL=custom_bert
```

### 2. Add New LLM Provider
```python
# services/reasoning/src/reasoning/llm/
class AnthropicProvider(LLMProvider):
    async def reason(self, prompt):
        # Claude API call
        return response

# Update config
LLM_PROVIDER=anthropic
```

### 3. Add New Action Policy
```python
# services/action-executor/src/action_executor/policies/
class ContextualPolicy(ActionPolicy):
    def recommend_action(self, context):
        # Custom logic based on user history, platform, etc.
        return Action(...)

# Update config
ACTION_POLICY=contextual
```

### 4. Add New Platform
```python
# services/event-ingestion/src/event_ingestion/connectors/
class RedditConnector(BasePlatformConnector):
    def normalize_event(self, webhook):
        # Reddit → Event translation
        return Event(...)

# Update platform enum + connector registry
```

---

## 🔐 Security Checklist

```
□ X-API-Key validation on all webhooks
□ JWT secret configured (services/shared/security/auth.py)
□ Database passwords changed from defaults
□ Redis password set
□ TLS enabled on external endpoints
□ Network policies restrict service access
□ Secrets stored in vault (not in code)
□ Input validation (Pydantic validators)
□ Rate limiting configured
□ Audit logging enabled
□ PII redaction in logs
□ Regular security scans (Trivy)
```

---

## 📈 Performance Tuning

```
# PostgreSQL
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB

# Redis
maxmemory = 4gb
maxmemory-policy allkeys-lru

# Python (Uvicorn)
workers = 4 * CPU_CORES
worker_class = uvicorn.workers.UvicornWorker

# Qdrant
vector_size = 1536 (match embedding model)
distance = "Cosine"
index_type = "Hnsw"
```

---

## 🐛 Troubleshooting Reference

| Problem | Check | Solution |
|---------|-------|----------|
| Event not processing | Redis stream pending | `redis-cli XLEN events:raw` |
| High latency | Service CPU | `kubectl top pods` |
| Model inference slow | Cache hit rate | Check Redis key hits |
| LLM errors | API quota | Check OpenAI billing |
| Queue backlog | Consumer lag | Scale consumer replicas |
| Database slow | Connections | Check `pg_stat_activity` |
| Vector search slow | Qdrant index | Rebuild with HNSW |

---

## 📚 File Navigation

```
SYSTEM_DESIGN.md
├─ When: Understanding "why" architecture decisions
├─ What: All components, contracts, data flow
└─ How: Implementation patterns explained

MONOREPO_STRUCTURE.md
├─ When: Planning code organization
├─ What: Complete directory tree
└─ How: File purposes explained

docker-compose.yml
├─ When: Getting up running locally
├─ What: All services + infrastructure
└─ How: Ready to use, just run

shared_models.py
├─ When: Understanding data contracts
├─ What: Core Pydantic models
└─ How: Copy to shared/safety_system/core/

event_ingestion_main.py
├─ When: Implementing services
├─ What: FastAPI service skeleton
└─ How: Template for all services

DEPLOYMENT_GUIDE.md
├─ When: Going to production
├─ What: Step-by-step deployment
└─ How: Local → Staging → Prod

IMPLEMENTATION_SUMMARY.md
├─ When: Assessing project scope
├─ What: Big picture + roadmap
└─ How: Next steps + timeline

QUICK_REFERENCE.md (this file!)
├─ When: You need a cheat sheet
├─ What: Quick lookup of everything
└─ How: Copy/paste ready templates
```

---

## ⏱️ Time to Production

| Milestone | Effort | Time |
|-----------|--------|------|
| Local dev working | Easy | 1-2 hours |
| Screening service | Medium | 1-2 days |
| Full stack working | Medium | 3-5 days |
| Kubernetes deploy | Medium | 2-3 days |
| Production ready | Hard | 1-2 weeks |

**Critical Path**: Event Ingestion → Screening → Action Executor  
**Can Parallelize**: Embedding/Reasoning services can be added later

---

## 💬 Key Terminology

| Term | Meaning |
|------|---------|
| **Harassment** | Temporal escalation pattern, not single incident |
| **Session** | Conversation thread (30-min TTL) |
| **Escalation Score** | Measure of increasing intensity over time |
| **Temporal Features** | Time-series characteristics (intervals, bursts) |
| **RAG** | Retrieve similar cases, Augment prompt, Generate answer |
| **Evidence Trace** | Why decision made (similar case, history, etc.) |
| **Graduated Action** | Warning → Timeout → Mute → Ban |
| **Consumer Lag** | How behind a stream consumer is |
| **Event Replay** | Re-process events from Redis (for debugging) |

---

## 🎓 Learning Path

1. **Day 1**: Read SYSTEM_DESIGN.md sections 1-4 (understand harassment as temporal process)
2. **Day 2**: Read MONOREPO_STRUCTURE.md + run `docker-compose up`
3. **Day 3**: Deploy first service using event_ingestion_main.py template
4. **Day 4**: Integrate with your platform
5. **Day 5**: Load test and tune
6. **Week 2**: Go to Kubernetes
7. **Week 3+**: Integrate LLM + fine-tune models

**Expected outcome**: Production deployment in 3-4 weeks from start

---

**Last updated:** January 22, 2026  
**Status:** Production-ready specification  
**Next action:** `docker-compose up` 🚀
