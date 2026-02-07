# services/event-ingestion/src/event_ingestion/main.py
"""
Event Ingestion Service - Production Implementation

Responsibility:
  - Accept events from multiple platforms (Discord, Slack, Web, etc.)
  - Async batch ingestion with backpressure handling
  - Input validation and normalization
  - Event deduplication
  - Redis Streams producer
  - Graceful degradation

Performance Targets:
  - Throughput: 10,000+ events/sec
  - Latency: P99 < 100ms
  - Availability: 99.9%
"""

import asyncio
import logging
import time
import json
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from uuid import uuid4
import hashlib

from fastapi import FastAPI, HTTPException, Header, Body, status
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import redis.asyncio.client as Redis
from pydantic import ValidationError
import opentelemetry.trace as trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
import prometheus_client as prom

# Shared library
from shared.safety_system.core.models import Event, EventType, Platform, EventMetadata


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


# ============ Metrics ============
from prometheus_client import REGISTRY

def get_or_create_counter(name, documentation, labelnames=()):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return prom.Counter(name, documentation, labelnames)

def get_or_create_gauge(name, documentation):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return prom.Gauge(name, documentation)

def get_or_create_histogram(name, documentation, buckets=()):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return prom.Histogram(name, documentation, buckets=buckets)

events_received = get_or_create_counter(
    'events_received_total',
    'Total events received',
    ['platform', 'event_type']
)

events_deduplicated = get_or_create_counter(
    'events_deduplicated_total',
    'Total duplicate events',
    ['platform']
)

events_queued = get_or_create_gauge(
    'events_queue_depth',
    'Current queue depth'
)

ingestion_latency = get_or_create_histogram(
    'ingestion_latency_ms',
    'Event ingestion latency',
    buckets=[10, 25, 50, 100, 250, 500, 1000]
)

validation_errors = get_or_create_counter(
    'validation_errors_total',
    'Validation errors',
    ['error_type']
)


# ============ Configuration ============

class Settings:
    """Service configuration"""

    def __init__(self):
        import os

        # Service
        self.service_name = "event-ingestion"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.port = int(os.getenv("PORT", "8001"))
        self.workers = int(os.getenv("WORKERS", "4"))
        self.max_body_size = int(os.getenv("MAX_BODY_SIZE", "1048576"))  # 1MB

        # Redis
        self.redis_host = os.getenv("REDIS_HOST", "redis")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "")

        # Ingestion
        self.output_stream = "events:raw"
        self.max_queue_depth = int(os.getenv("MAX_QUEUE_DEPTH", "100000"))
        self.batch_timeout_ms = int(os.getenv("BATCH_TIMEOUT_MS", "100"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "500"))

        # Deduplication
        self.dedup_window_seconds = int(os.getenv("DEDUP_WINDOW_SECONDS", "3600"))  # 1 hour
        self.dedup_cache_key = "event:dedup"

        # Validation
        self.min_content_length = int(os.getenv("MIN_CONTENT_LENGTH", "1"))
        self.max_content_length = int(os.getenv("MAX_CONTENT_LENGTH", "10000"))
        self.require_author = os.getenv("REQUIRE_AUTHOR", "true").lower() == "true"

        # Auth
        self.valid_api_keys = set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else {"test-key"}
        self.valid_platforms = {p.value for p in Platform}

        # Backpressure
        self.backpressure_enabled = os.getenv("BACKPRESSURE_ENABLED", "true").lower() == "true"
        self.backpressure_threshold = float(os.getenv("BACKPRESSURE_THRESHOLD", "0.8"))


settings = Settings()


# ============ Event Validation ============

class EventValidator:
    """
    Validates and normalizes incoming events.

    Catches:
    - Missing required fields
    - Invalid content length
    - Invalid platforms
    - Malformed data
    - Platform-specific violations
    """

    @staticmethod
    def validate(raw_data: Dict[str, Any]) -> Event:
        """
        Validate and normalize event.

        Raises:
            ValueError: If validation fails

        Returns:
            Validated Event
        """

        # Required fields
        required_fields = ['event_id', 'event_type', 'platform', 'content', 'user_id', 'author_id']
        missing = [f for f in required_fields if f not in raw_data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Platform validation
        platform = raw_data.get('platform', '').lower()
        if platform not in settings.valid_platforms:
            raise ValueError(f"Invalid platform: {platform}")

        # Event type validation
        event_type = raw_data.get('event_type', '').lower()
        try:
            EventType(event_type)
        except ValueError:
            raise ValueError(f"Invalid event_type: {event_type}")

        # Content validation
        content = raw_data.get('content', '').strip()
        if len(content) < settings.min_content_length:
            raise ValueError(f"Content too short (min: {settings.min_content_length})")
        if len(content) > settings.max_content_length:
            raise ValueError(f"Content too long (max: {settings.max_content_length})")

        # Author validation
        if settings.require_author and not raw_data.get('author_id'):
            raise ValueError("author_id required")

        # IDs validation (should be non-empty strings)
        for id_field in ['user_id', 'author_id', 'channel_id']:
            if id_field in raw_data:
                value = raw_data.get(id_field, '').strip()
                if value and len(value) > 500:  # Reasonable max ID length
                    raise ValueError(f"{id_field} too long")

        # Create event
        event = Event(
            event_id=raw_data['event_id'],
            event_type=event_type,
            platform=platform,
            channel_id=raw_data.get('channel_id', ''),
            user_id=raw_data.get('user_id', ''),
            author_id=raw_data.get('author_id', ''),
            content=content,
            metadata=EventMetadata(
                **raw_data.get('metadata', {})
            ),
            created_at=datetime.fromisoformat(
                raw_data.get('created_at', datetime.utcnow().isoformat())
            )
        )

        return event


# ============ Deduplication ============

class Deduplicator:
    """
    Detects duplicate events using content hash.

    Strategy:
    - Hash event content
    - Check if seen in window (Redis set with TTL)
    - Mark as seen if new
    """

    def __init__(self, redis: Redis):
        self.redis = redis

    def _compute_hash(self, event: Event) -> str:
        """Compute event content hash"""
        content = f"{event.platform}:{event.author_id}:{event.content}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def is_duplicate(self, event: Event) -> bool:
        """Check if event is duplicate"""
        hash_val = self._compute_hash(event)
        key = f"{settings.dedup_cache_key}:{hash_val}"

        # Check existence
        exists = await self.redis.exists(key)

        if not exists:
            # Mark as seen
            await self.redis.setex(
                key,
                settings.dedup_window_seconds,
                "1"
            )

        return bool(exists)


# ============ Backpressure Handling ============

class BackpressureManager:
    """
    Manages queue depth and applies backpressure.

    When queue depth > threshold:
    - Return 503 (Service Unavailable) to clients
    - Clients should retry with exponential backoff
    - Prevents system overload
    """

    def __init__(self, redis: Redis):
        self.redis = redis

    async def check_capacity(self, stream_name: str) -> bool:
        """Check if queue has capacity"""
        if not settings.backpressure_enabled:
            return True

        depth = await self.redis.xlen(stream_name)
        capacity_ok = depth < settings.max_queue_depth

        ratio = depth / settings.max_queue_depth if settings.max_queue_depth > 0 else 0

        # Log when approaching threshold
        if ratio > settings.backpressure_threshold:
            logger.warning(
                f"Queue depth: {depth}/{settings.max_queue_depth} ({ratio:.1%})"
            )

        return capacity_ok


# ============ Event Ingestion Engine ============

class IngestionEngine:
    """
    Main ingestion engine.

    Flow:
    1. Receive event
    2. Validate
    3. Check duplicate
    4. Check backpressure
    5. Add to Redis Stream
    6. Respond to client
    """

    def __init__(self, redis: redis.Redis):
        self.redis = redis
        self.validator = EventValidator()
        self.deduplicator = Deduplicator(redis)
        self.backpressure_mgr = BackpressureManager(redis)

    async def ingest(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest single event.

        Raises:
            ValueError: Validation error
            RuntimeError: Backpressure exceeded

        Returns:
            {
                "event_id": str,
                "status": "queued" | "duplicate",
                "queue_position": int,
                "ingestion_time_ms": float
            }
        """

        start_time = time.perf_counter()

        with tracer.start_as_current_span("ingest_event") as span:
            try:
                # 1. Validate
                event = self.validator.validate(raw_data)
                span.set_attribute("event_id", event.event_id)
                span.set_attribute("platform", event.platform)
                span.set_attribute("content_length", len(event.content))

                # 2. Check duplicate
                is_dup = await self.deduplicator.is_duplicate(event)
                if is_dup:
                    span.set_attribute("duplicate", True)
                    events_deduplicated.labels(platform=event.platform).inc()
                    return {
                        "event_id": event.event_id,
                        "status": "duplicate",
                        "ingestion_time_ms": (time.perf_counter() - start_time) * 1000
                    }

                # 3. Check backpressure
                has_capacity = await self.backpressure_mgr.check_capacity(settings.output_stream)
                if not has_capacity:
                    span.set_attribute("backpressure", True)
                    raise RuntimeError("Queue full (backpressure)")

                # 4. Add to stream
                stream_data = event.dict()
                stream_data["metadata"] = json.dumps(stream_data.get("metadata", {}))

                position = await self.redis.xadd(
                    settings.output_stream,
                    stream_data
                )

                # 5. Respond
                ingestion_ms = (time.perf_counter() - start_time) * 1000

                events_received.labels(
                    platform=event.platform,
                    event_type=event.event_type
                ).inc()
                ingestion_latency.observe(ingestion_ms)

                logger.debug(
                    f"Ingested {event.event_id} ({event.platform}) in {ingestion_ms:.1f}ms"
                )

                return {
                    "event_id": event.event_id,
                    "status": "queued",
                    "position": position.decode() if isinstance(position, bytes) else position,
                    "ingestion_time_ms": ingestion_ms
                }

            except ValueError as e:
                span.record_exception(e)
                validation_errors.labels(error_type="validation").inc()
                logger.warning(f"Validation error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

            except RuntimeError as e:
                span.record_exception(e)
                if "backpressure" in str(e).lower():
                    logger.warning(f"Backpressure: {e}")
                    raise HTTPException(status_code=503, detail=str(e))
                raise HTTPException(status_code=500, detail=str(e))

            except Exception as e:
                span.record_exception(e)
                logger.error(f"Ingestion error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Internal server error")


# ============ FastAPI App ============

redis_client: Optional[redis.Redis] = None
ingestion_engine: Optional[IngestionEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""

    global redis_client, ingestion_engine

    # Startup
    logger.info(f"Starting {settings.service_name}")

    # Connect to Redis
    redis_client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password or None,
        decode_responses=True
    )
    await redis_client.ping()

    ingestion_engine = IngestionEngine(redis_client)

    logger.info(f"{settings.service_name} ready")

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.service_name}")
    redis_client.close()
    await redis_client.wait_closed()


app = FastAPI(
    title="Event Ingestion Service",
    description="High-performance event ingestion with validation and backpressure",
    version="1.0.0",
    lifespan=lifespan
)

FastAPIInstrumentor.instrument_app(app)
RedisInstrumentor().instrument()


# ============ Health Checks ============

@app.get("/health")
async def health_check():
    """Liveness probe"""
    return {
        "status": "ok",
        "service": settings.service_name,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/ready")
async def readiness_check():
    """Readiness probe"""
    try:
        if not redis_client:
            raise RuntimeError("Redis not initialized")

        await redis_client.ping()

        return {
            "status": "ready",
            "redis": "ok",
            "max_queue_depth": settings.max_queue_depth
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


# ============ API Endpoints ============

@app.post("/v1/events/ingest")
async def ingest_event(
    body: Dict[str, Any] = Body(...),
    api_key: str = Header(None)
) -> Dict[str, Any]:
    """
    Ingest single event.

    Request:
    {
        "event_id": "unique-id-123",
        "event_type": "message_created",
        "platform": "discord",
        "channel_id": "channel-123",
        "user_id": "user-123",
        "author_id": "author-123",
        "content": "This is a test message",
        "created_at": "2025-01-22T10:30:00Z",
        "metadata": {
            "user_reputation": 0.8,
            "author_reputation": 0.6
        }
    }

    Response (success):
    {
        "event_id": "unique-id-123",
        "status": "queued",
        "position": "1234567890-0",
        "ingestion_time_ms": 15.3
    }

    Response (duplicate):
    {
        "event_id": "unique-id-123",
        "status": "duplicate",
        "ingestion_time_ms": 2.1
    }

    Response (backpressure, HTTP 503):
    {
        "detail": "Queue full (backpressure)"
    }
    """

    # Validate API key
    if api_key not in settings.valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not ingestion_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return await ingestion_engine.ingest(body)


@app.post("/v1/events/ingest-batch")
async def ingest_batch(
    body: Dict[str, Any] = Body(...),
    api_key: str = Header(None)
) -> Dict[str, Any]:
    """
    Ingest batch of events.

    Request:
    {
        "events": [
            {
                "event_id": "event-1",
                ...
            },
            {
                "event_id": "event-2",
                ...
            }
        ]
    }

    Response:
    {
        "total": 2,
        "queued": 2,
        "duplicates": 0,
        "errors": 0,
        "results": [
            {"event_id": "event-1", "status": "queued"},
            {"event_id": "event-2", "status": "queued"}
        ],
        "batch_time_ms": 25.4
    }
    """

    if api_key not in settings.valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not ingestion_engine:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start_time = time.perf_counter()
    events = body.get("events", [])

    # Limit batch size
    if len(events) > settings.batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large (max: {settings.batch_size})"
        )

    # Ingest in parallel
    tasks = [ingestion_engine.ingest(event) for event in events]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    queued = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "queued")
    duplicates = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "duplicate")
    errors = sum(1 for r in results if isinstance(r, Exception))

    return {
        "total": len(events),
        "queued": queued,
        "duplicates": duplicates,
        "errors": errors,
        "results": results,
        "batch_time_ms": (time.perf_counter() - start_time) * 1000
    }


@app.get("/v1/metrics")
async def get_metrics():
    """Return service metrics"""
    try:
        if not redis_client:
            raise RuntimeError("Redis not initialized")

        queue_depth = await redis_client.xlen(settings.output_stream)

        return {
            "service": settings.service_name,
            "queue_depth": queue_depth,
            "max_queue_depth": settings.max_queue_depth,
            "queue_capacity_percent": (queue_depth / settings.max_queue_depth * 100) if settings.max_queue_depth > 0 else 0,
            "dedup_window_seconds": settings.dedup_window_seconds,
            "backpressure_threshold": settings.backpressure_threshold,
            "backpressure_enabled": settings.backpressure_enabled
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=settings.log_level)

    uvicorn.run(
        "src.event_ingestion.main:app",
        host="0.0.0.0",
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower()
    )
