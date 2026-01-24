# Monorepo Directory Structure
## AI-Powered Personal Safety Infrastructure

```
safety-system/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── pyproject.toml               # Python project config + dependencies
├── poetry.lock                  # Locked dependencies
│
├── docker-compose.yml           # Local dev environment
├── docker-compose.prod.yml      # Production-like environment
│
├── Dockerfile                   # Multi-stage Python image
├── Dockerfile.worker            # Async worker image
│
├── k8s/                         # Kubernetes manifests
│   ├── namespace.yaml
│   ├── postgres-pvc.yaml
│   ├── redis-deployment.yaml
│   ├── qdrant-deployment.yaml
│   ├── kustomization.yaml
│   │
│   └── services/
│       ├── event-ingestion/
│       │   ├── deployment.yaml
│       │   ├── service.yaml
│       │   ├── hpa.yaml
│       │   └── ingress.yaml
│       ├── risk-screening/
│       │   ├── deployment.yaml
│       │   ├── service.yaml
│       │   ├── hpa.yaml
│       │   └── configmap.yaml
│       ├── embedding-context/
│       │   ├── deployment.yaml
│       │   ├── service.yaml
│       │   └── hpa.yaml
│       ├── reasoning/
│       │   ├── deployment.yaml
│       │   ├── service.yaml
│       │   └── hpa.yaml
│       ├── action-executor/
│       │   ├── deployment.yaml
│       │   ├── service.yaml
│       │   └── hpa.yaml
│       ├── session-manager/
│       │   ├── deployment.yaml
│       │   └── service.yaml
│       └── reporting/
│           ├── deployment.yaml
│           ├── service.yaml
│           └── cronjob.yaml
│
├── helm/                        # Helm charts (alternative to kustomize)
│   └── safety-system/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values-prod.yaml
│       └── templates/
│
├── shared/                      # Shared libraries (imported by all services)
│   ├── __init__.py
│   ├── pyproject.toml
│   ├── requirements.txt
│   │
│   ├── safety_system/
│   │   ├── __init__.py
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── models.py          # Pydantic models (Event, Action, Risk, etc.)
│   │   │   ├── schemas.py         # API schemas
│   │   │   ├── exceptions.py      # Custom exceptions
│   │   │   ├── constants.py       # Constants (risk categories, action types)
│   │   │   └── types.py           # Type definitions
│   │   │
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── connection.py      # Database connection pooling
│   │   │   ├── models.py          # SQLAlchemy ORM models
│   │   │   ├── migrations.py      # Alembic migrations
│   │   │   └── seeds.py           # Database seeding
│   │   │
│   │   ├── cache/
│   │   │   ├── __init__.py
│   │   │   ├── redis_client.py    # Redis wrapper
│   │   │   ├── session.py         # Session state
│   │   │   └── keys.py            # Redis key naming conventions
│   │   │
│   │   ├── vector/
│   │   │   ├── __init__.py
│   │   │   ├── qdrant_client.py   # Qdrant wrapper
│   │   │   ├── embedding.py       # Embedding models (interface + providers)
│   │   │   ├── search.py          # Vector search utilities
│   │   │   └── payload.py         # Payload schema helpers
│   │   │
│   │   ├── streams/
│   │   │   ├── __init__.py
│   │   │   ├── redis_streams.py   # Redis Streams wrapper
│   │   │   ├── consumer.py        # Consumer group abstraction
│   │   │   ├── producer.py        # Event producer
│   │   │   └── retry_policy.py    # DLQ and retry logic
│   │   │
│   │   ├── security/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py            # JWT, API key validation
│   │   │   ├── rbac.py            # Role-based access control
│   │   │   └── crypto.py          # Encryption utilities
│   │   │
│   │   ├── observability/
│   │   │   ├── __init__.py
│   │   │   ├── logging.py         # Structured logging
│   │   │   ├── metrics.py         # Prometheus metrics
│   │   │   ├── tracing.py         # OpenTelemetry tracing
│   │   │   └── health.py          # Health check helpers
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── retry.py           # Exponential backoff
│   │       ├── validation.py      # Input validation
│   │       ├── serialization.py   # JSON/pickle helpers
│   │       └── timing.py          # Performance timing
│   │
│   └── tests/                     # Shared test fixtures
│       ├── __init__.py
│       ├── conftest.py            # pytest fixtures
│       ├── factories.py           # Factory patterns for test objects
│       └── mocks.py               # Mock services
│
├── services/
│   │
│   ├── event-ingestion/
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   │
│   │   ├── src/
│   │   │   └── event_ingestion/
│   │   │       ├── __init__.py
│   │   │       ├── main.py             # FastAPI app entry point
│   │   │       ├── config.py           # Service config
│   │   │       ├── routes.py           # API endpoints
│   │   │       │
│   │   │       ├── handlers/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── event_handler.py
│   │   │       │   └── webhook_handler.py
│   │   │       │
│   │   │       ├── connectors/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── base.py
│   │   │       │   ├── discord.py
│   │   │       │   ├── twitch.py
│   │   │       │   └── web_chat.py
│   │   │       │
│   │   │       ├── schema_normalizer.py
│   │   │       ├── deduplicator.py
│   │   │       └── producer.py
│   │   │
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_routes.py
│   │       ├── test_handlers.py
│   │       ├── test_connectors.py
│   │       └── test_integration.py
│   │
│   ├── risk-screening/
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   │
│   │   ├── src/
│   │   │   └── risk_screening/
│   │   │       ├── __init__.py
│   │   │       ├── main.py
│   │   │       ├── config.py
│   │   │       ├── routes.py
│   │   │       │
│   │   │       ├── models/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── base.py             # Abstract RiskClassifier
│   │   │       │   ├── tinybert.py         # Fast BERT variant
│   │   │       │   ├── distilbert.py       # Balanced BERT variant
│   │   │       │   └── ensemble.py         # Voting ensemble
│   │   │       │
│   │   │       ├── rules/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── pattern_matching.py # Regex rules
│   │   │       │   ├── heuristics.py       # Domain heuristics
│   │   │       │   └── context_rules.py    # User history rules
│   │   │       │
│   │   │       ├── screening.py
│   │   │       └── producer.py
│   │   │
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_screening.py
│   │       ├── test_models.py
│   │       └── test_rules.py
│   │
│   ├── embedding-context/
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   │
│   │   ├── src/
│   │   │   └── embedding_context/
│   │   │       ├── __init__.py
│   │   │       ├── main.py
│   │   │       ├── config.py
│   │   │       ├── routes.py
│   │   │       │
│   │   │       ├── embedding_service.py
│   │   │       ├── retrieval_service.py
│   │   │       ├── session_aggregator.py
│   │   │       └── consumer.py            # Redis Stream consumer
│   │   │
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_embedding.py
│   │       ├── test_retrieval.py
│   │       └── test_session.py
│   │
│   ├── reasoning/
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   │
│   │   ├── src/
│   │   │   └── reasoning/
│   │   │       ├── __init__.py
│   │   │       ├── main.py
│   │   │       ├── config.py
│   │   │       ├── routes.py
│   │   │       │
│   │   │       ├── llm/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── base_provider.py    # Abstract LLMProvider
│   │   │       │   ├── openai_provider.py
│   │   │       │   ├── local_provider.py   # Ollama/Llama
│   │   │       │   └── prompt_builder.py
│   │   │       │
│   │   │       ├── rag_pipeline.py
│   │   │       ├── reasoning_engine.py
│   │   │       ├── fallback_rules.py       # Rule-based fallback if LLM fails
│   │   │       ├── consumer.py
│   │   │       └── producer.py
│   │   │
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_reasoning.py
│   │       ├── test_llm_providers.py
│   │       └── test_rag_pipeline.py
│   │
│   ├── action-executor/
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   │
│   │   ├── src/
│   │   │   └── action_executor/
│   │   │       ├── __init__.py
│   │   │       ├── main.py
│   │   │       ├── config.py
│   │   │       ├── routes.py
│   │   │       │
│   │   │       ├── policies/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── base_policy.py      # Abstract ActionPolicy
│   │   │       │   ├── graduated_policy.py # Warning → Timeout → Ban
│   │   │       │   └── severity_policy.py  # Risk score based
│   │   │       │
│   │   │       ├── actions/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── base_action.py
│   │   │       │   ├── warning.py
│   │   │       │   ├── timeout.py
│   │   │       │   ├── mute.py
│   │   │       │   └── ban.py
│   │   │       │
│   │   │       ├── executor.py
│   │   │       ├── consumer.py
│   │   │       ├── audit_logger.py
│   │   │       └── expiry_manager.py       # CronJob handler for timeouts
│   │   │
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_executor.py
│   │       ├── test_policies.py
│   │       └── test_actions.py
│   │
│   ├── session-manager/
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   │
│   │   ├── src/
│   │   │   └── session_manager/
│   │   │       ├── __init__.py
│   │   │       ├── main.py
│   │   │       ├── config.py
│   │   │       ├── routes.py
│   │   │       │
│   │   │       ├── session.py             # Session lifecycle
│   │   │       ├── temporal_features.py   # Feature extraction
│   │   │       ├── burst_detector.py      # Burst detection
│   │   │       ├── escalation_detector.py # Escalation analysis
│   │   │       ├── consumer.py
│   │   │       └── aggregator.py          # Sliding window aggregation
│   │   │
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_session.py
│   │       ├── test_temporal.py
│   │       └── test_detectors.py
│   │
│   └── reporting/
│       ├── README.md
│       ├── pyproject.toml
│       ├── requirements.txt
│       │
│       ├── src/
│       │   └── reporting/
│       │       ├── __init__.py
│       │       ├── main.py
│       │       ├── config.py
│       │       ├── routes.py
│       │       │
│       │       ├── aggregators/
│       │       │   ├── __init__.py
│       │       │   ├── daily_aggregator.py
│       │       │   ├── hourly_aggregator.py
│       │       │   └── custom_queries.py
│       │       │
│       │       ├── analytics.py
│       │       ├── metrics_collector.py
│       │       └── jobs/
│       │           ├── __init__.py
│       │           ├── daily_report.py     # Arq background job
│       │           └── alert_checker.py    # Alert thresholds
│       │
│       └── tests/
│           ├── __init__.py
│           ├── test_aggregators.py
│           └── test_analytics.py
│
├── scripts/
│   ├── __init__.py
│   ├── bootstrap_db.py          # Initialize schema
│   ├── seed_data.py             # Load test data
│   ├── load_test.py             # Locust/K6 load testing
│   ├── migrate.py               # Alembic wrapper
│   ├── vector_index_rebuild.py  # Qdrant index management
│   └── audit_export.py          # Export audit logs
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Global pytest fixtures
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_event_to_action.py    # End-to-end flow
│   │   ├── test_temporal_analysis.py
│   │   └── test_rag_reasoning.py
│   └── performance/
│       ├── __init__.py
│       ├── test_screening_latency.py
│       └── test_throughput.py
│
├── docs/
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── DEPLOYMENT.md
│   ├── DEVELOPMENT.md
│   ├── SECURITY.md
│   └── TROUBLESHOOTING.md
│
├── monitoring/
│   ├── prometheus.yml           # Prometheus scrape config
│   ├── grafana-dashboards/
│   │   ├── safety_system_overview.json
│   │   ├── service_latency.json
│   │   └── queue_depth.json
│   ├── alerts.yml               # AlertManager rules
│   └── jaeger-config.yml        # Distributed tracing
│
├── .github/
│   ├── workflows/
│   │   ├── test.yml             # CI: unit + integration tests
│   │   ├── lint.yml             # CI: code quality
│   │   ├── build.yml            # Build Docker images
│   │   ├── security-scan.yml    # Trivy security scanning
│   │   └── deploy.yml           # CD: deploy to Kubernetes
│   └── CODEOWNERS
│
├── .dockerignore
├── .eslintignore
├── .pre-commit-config.yaml      # Pre-commit hooks (black, mypy, isort)
├── Makefile                      # Development shortcuts
│
└── CONTRIBUTING.md
```

---

## Key Design Patterns

### 1. **Shared Library Pattern**
- `shared/safety_system/` contains reusable code
- All services import from `safety_system` package
- Central location for models, DB connection, caching, etc.
- Reduces duplication, ensures consistency

### 2. **Service Autonomy**
- Each service has its own FastAPI app, routes, tests
- Services communicate via Redis Streams (async)
- No direct service-to-service calls (except critical path)
- Each service owns its domain model + business logic

### 3. **Plugin Architecture**
- `models/` in risk-screening: swap ML classifiers without changing API
- `llm/providers.py` in reasoning: OpenAI ↔ Local LLM
- `policies/` in action-executor: customize mitigation strategies
- `connectors/` in event-ingestion: add new platforms

### 4. **Layered Code Organization**
```
Service Structure:
├── routes.py        (FastAPI endpoints)
├── main.py          (app initialization)
├── config.py        (environment config)
├── [domain]/
│   ├── models.py    (business logic)
│   ├── service.py   (use cases)
│   └── repository.py (data access)
└── consumer.py      (Redis Stream worker)
```

### 5. **Async-First Design**
- All I/O is async (FastAPI, Redis, PostgreSQL, vector DB)
- Background workers use Arq (async task queue)
- No blocking calls in request path

---

## Configuration Strategy

### Environment Variables

```bash
# Common (all services)
LOG_LEVEL=INFO
ENVIRONMENT=development|staging|production
SENTRY_DSN=https://...

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/safety

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=secret (optional)

# Authentication
JWT_SECRET_KEY=...
API_KEYS_VALID=key1,key2,key3

# LLM (reasoning service)
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai|local
LOCAL_LLM_ENDPOINT=http://ollama:11434

# Risk Screening
ML_MODEL_PATH=/models/tinybert.onnx
SCREENING_THRESHOLD=0.7

# Tracing
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
OTEL_EXPORTER_OTLP_HEADERS=...
```

---

## Testing Strategy

### Unit Tests
- Per-service, isolated from external dependencies
- Mock Redis, PostgreSQL, LLM, vector DB
- Fast execution (< 100ms per test)

### Integration Tests
- Services + infrastructure (Docker containers)
- Docker Compose test environment
- Full data flow validation

### Performance Tests
- Load test with Locust/K6
- Latency percentile tracking (p99, p95, p50)
- Throughput benchmarking

### E2E Tests
- Event ingestion → screening → reasoning → action execution
- Verify state changes in database
- Validate audit trail

---

## Deployment Stages

1. **Local Development**: Docker Compose, fast iteration
2. **Staging**: Kubernetes cluster, production-like environment
3. **Production**: Multi-AZ Kubernetes, managed PostgreSQL + Redis

---

## Summary

This structure provides:
✅ **Clear separation of concerns** (7 independent services)
✅ **Code reuse** (shared library prevents duplication)
✅ **Rapid iteration** (plugin architecture for models/policies)
✅ **Testability** (isolated unit tests, integration test suite)
✅ **Observability** (structured logging, metrics, tracing)
✅ **Scalability** (stateless services, horizontal scaling)
✅ **Maintainability** (consistent conventions, documented patterns)
