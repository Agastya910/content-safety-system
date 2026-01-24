# services/risk-screening/src/risk_screening/main.py
"""
Risk Screening Service - Production Implementation

Responsibility:
  - Ultra-fast risk classification (< 50ms)
  - Text embedding (e5-small-v2)
  - FAISS/Qdrant vector similarity search
  - Behavioral heuristics scoring
  - Batch processing from Redis Streams

Performance Targets:
  - P99 latency: < 50ms
  - Throughput: 5000+ events/sec
  - Model inference: < 30ms per event
"""

import asyncio
import logging
import time
import json
import numpy as np
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4
import re
from collections import Counter

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.responses import JSONResponse
import aioredis
from aioredis import Redis
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import opentelemetry.trace as trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

# Shared library
from safety_system.core.models import (
    Event, RiskPrediction, RiskCategory, RiskFlag
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


# ============ Configuration ============

class Settings:
    """Service configuration"""

    def __init__(self):
        import os

        # Service
        self.service_name = "risk-screening"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "4"))

        # Redis
        self.redis_host = os.getenv("REDIS_HOST", "redis")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "")

        # Screening
        self.screening_model = os.getenv("SCREENING_MODEL", "e5-small-v2")
        self.screening_threshold = float(os.getenv("SCREENING_THRESHOLD", "0.7"))
        self.embedding_cache_ttl = int(os.getenv("EMBEDDING_CACHE_TTL", "604800"))  # 7 days

        # Model
        self.model_batch_size = int(os.getenv("MODEL_BATCH_SIZE", "32"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp32 = os.getenv("USE_FP32", "false").lower() == "true"

        # Redis Streams
        self.input_stream = "events:raw"
        self.output_stream_high = "events:reasoning_queue"  # risk > threshold
        self.output_stream_low = "events:low_risk"  # risk <= threshold
        self.consumer_group = "risk-screening-group"
        self.consumer_name = f"risk-screening-{os.getenv('HOSTNAME', 'local')}"
        self.batch_timeout_ms = int(os.getenv("BATCH_TIMEOUT_MS", "500"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "100"))

        # Auth
        self.valid_api_keys = set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else {"test-key"}


settings = Settings()


# ============ Embedding Model ============

class EmbeddingModel:
    """
    Manages sentence embedding model (e5-small-v2).

    e5-small-v2 is optimized for:
    - Small model size (33M params)
    - Fast inference (< 20ms per text)
    - Good quality embeddings (384-dim)
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.screening_model}")
        self.model = SentenceTransformer(
            settings.screening_model,
            device=settings.device
        )

        # Reduce model to FP32 if configured (faster but less accurate)
        if not settings.use_fp32:
            self.model.max_seq_length = 256  # Truncate long sequences

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded (dim={self.embedding_dim}, device={settings.device})")

    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed batch of texts.

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self.model.encode,
            texts,
            False,  # convert_to_numpy=True
            True    # normalize_embeddings=True
        )
        return np.array(embeddings, dtype=np.float32)


# ============ Behavioral Heuristics ============

class BehavioralHeuristics:
    """
    Fast rule-based scoring for harassment indicators.

    Complements ML model for quick detection of obvious patterns.
    """

    # Patterns
    CAPS_THRESHOLD = 0.8  # > 80% caps
    REPEATED_CHARS = r'(.)\1{4,}'  # 5+ repeated chars
    PUNCTUATION_EXCESSIVE = r'[!?]{3,}'  # 3+ repeated punctuation

    # Toxic words (simplified, real system would be larger)
    OFFENSIVE_KEYWORDS = {
        'trash', 'terrible', 'awful', 'stupid', 'idiot', 'fool',
        'bad', 'hate', 'despise', 'pathetic', 'loser'
    }

    # Targeting patterns
    TARGETING_KEYWORDS = {
        'you are', 'you\\'re', 'your', 'you all', 'y\'all'
    }

    @staticmethod
    def score(text: str) -> Tuple[float, List[str]]:
        """
        Calculate heuristic risk score (0-1).

        Returns:
            (score, flags)
        """
        score = 0.0
        flags = []

        # Lowercase for analysis
        text_lower = text.lower()

        # ALL CAPS check
        if len(text) > 10:  # Ignore short texts
            letter_count = sum(1 for c in text if c.isalpha())
            if letter_count > 0:
                caps_ratio = sum(1 for c in text if c.isupper()) / letter_count
                if caps_ratio > BehavioralHeuristics.CAPS_THRESHOLD:
                    score += 0.2
                    flags.append(RiskFlag.ALL_CAPS_SPAM)

        # Repeated punctuation
        if re.search(BehavioralHeuristics.PUNCTUATION_EXCESSIVE, text):
            score += 0.15
            flags.append(RiskFlag.REPEATED_PUNCTUATION)

        # Repeated characters
        if re.search(BehavioralHeuristics.REPEATED_CHARS, text):
            score += 0.15
            flags.append(RiskFlag.REPEATED_PUNCTUATION)

        # Offensive language
        words = text_lower.split()
        offensive_count = sum(1 for w in words if w in BehavioralHeuristics.OFFENSIVE_KEYWORDS)
        if offensive_count > 0:
            score += min(0.3, offensive_count * 0.1)
            flags.append(RiskFlag.OFFENSIVE_LANGUAGE)

        # Targeting language
        targeting_count = sum(1 for keyword in BehavioralHeuristics.TARGETING_KEYWORDS if keyword in text_lower)
        if targeting_count > 0:
            score += 0.2
            flags.append(RiskFlag.TARGETING_BEHAVIOR)

        # Cap score at 1.0
        return min(score, 1.0), flags


# ============ FAISS Vector Store ============

class FAISSVectorStore:
    """
    In-memory vector index for similarity search.

    Used for finding similar past harassment patterns.
    In production, this would be backed by Qdrant for persistence.
    """

    def __init__(self, embedding_dim: int = 384):
        # Use Flat (exact search) for accuracy
        # In production with millions of vectors, use IVF-PQ or HNSW
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.embeddings: Dict[int, np.ndarray] = {}
        self.event_ids: Dict[int, str] = {}
        self.id_counter = 0
        self.lock = asyncio.Lock()

    async def add(self, event_id: str, embedding: np.ndarray):
        """Add embedding to index"""
        async with self.lock:
            # Ensure proper shape
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)

            # Add to FAISS
            self.index.add(np.asarray(embedding, dtype=np.float32))
            self.embeddings[self.id_counter] = embedding
            self.event_ids[self.id_counter] = event_id
            self.id_counter += 1

    async def search(self, embedding: np.ndarray, k: int = 5) -> List[str]:
        """
        Search for k most similar embeddings.

        Returns:
            List of event_ids
        """
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)

        distances, indices = self.index.search(
            np.asarray(embedding, dtype=np.float32),
            min(k, self.index.ntotal)
        )

        # Return event IDs of nearest neighbors
        results = []
        for idx in indices[0]:
            if idx >= 0 and idx in self.event_ids:
                results.append(self.event_ids[idx])

        return results


# ============ Risk Screener ============

class RiskScreener:
    """
    Main screening engine combining:
    - ML embeddings + similarity search
    - Behavioral heuristics
    - Batch processing
    """

    def __init__(self, embedding_model: EmbeddingModel, vector_store: FAISSVectorStore, redis: Redis):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.redis = redis

    async def screen(self, event: Event, use_cache: bool = True) -> RiskPrediction:
        """
        Screen event for harassment risk.

        Process:
        1. Check embedding cache
        2. Generate/retrieve embedding
        3. Compute heuristic score
        4. Combine scores
        5. Search similar events

        Returns:
            RiskPrediction with risk_score, category, flags
        """

        start_time = time.perf_counter()

        with tracer.start_as_current_span("screen_event") as span:
            span.set_attribute("event_id", event.event_id)

            try:
                # 1. Try cache
                embedding = None
                if use_cache:
                    cached_embedding = await self.redis.get(f"embedding:{event.event_id}")
                    if cached_embedding:
                        embedding = np.frombuffer(
                            bytes.fromhex(cached_embedding),
                            dtype=np.float32
                        )
                        span.set_attribute("cache_hit", True)

                # 2. Generate embedding if not cached
                if embedding is None:
                    embedding_np = await self.embedding_model.embed([event.content])
                    embedding = embedding_np[0]

                    # Cache embedding
                    if use_cache:
                        embedding_hex = embedding.tobytes().hex()
                        await self.redis.set(
                            f"embedding:{event.event_id}",
                            embedding_hex,
                            expire=settings.embedding_cache_ttl
                        )

                    span.set_attribute("cache_hit", False)

                # 3. Behavioral heuristics score
                heuristic_score, heuristic_flags = BehavioralHeuristics.score(event.content)

                # 4. Embedding-based scoring (future: could use reference embeddings)
                embedding_score = 0.0  # Placeholder

                # 5. Combine scores (weighted)
                combined_score = 0.6 * heuristic_score + 0.4 * embedding_score

                # 6. Classify risk
                if combined_score > 0.85:
                    risk_category = RiskCategory.TARGETED_HARASSMENT
                elif combined_score > 0.70:
                    risk_category = RiskCategory.TOXIC
                elif combined_score > 0.50:
                    risk_category = RiskCategory.SPAM
                else:
                    risk_category = RiskCategory.LOW_RISK

                # 7. Search similar events (async)
                similar_events = await self.vector_store.search(embedding, k=3)

                # Calculate latency
                screening_time_ms = (time.perf_counter() - start_time) * 1000

                # Create prediction
                prediction = RiskPrediction(
                    event_id=event.event_id,
                    risk_score=combined_score,
                    risk_category=risk_category,
                    confidence=max(heuristic_score, embedding_score),
                    flags=[str(f) for f in heuristic_flags],
                    details={
                        "heuristic_score": float(heuristic_score),
                        "embedding_score": float(embedding_score),
                        "similar_events": similar_events
                    },
                    screening_time_ms=int(screening_time_ms)
                )

                # Set span attributes
                span.set_attribute("risk_score", combined_score)
                span.set_attribute("risk_category", risk_category.value)
                span.set_attribute("screening_time_ms", int(screening_time_ms))
                span.set_attribute("flag_count", len(heuristic_flags))

                logger.debug(
                    f"Screened {event.event_id}: score={combined_score:.2f}, "
                    f"category={risk_category.value}, time={screening_time_ms:.2f}ms"
                )

                return prediction

            except Exception as e:
                logger.error(f"Screening error for {event.event_id}: {e}", exc_info=True)
                span.record_exception(e)

                # Fallback to safe score
                return RiskPrediction(
                    event_id=event.event_id,
                    risk_score=0.5,
                    risk_category=RiskCategory.SPAM,
                    confidence=0.3,
                    screening_time_ms=int((time.perf_counter() - start_time) * 1000)
                )


# ============ Redis Stream Consumer ============

class StreamConsumer:
    """
    Consumes events from Redis Streams and screens them.
    Routes to appropriate output stream based on risk score.
    """

    def __init__(
        self,
        redis: Redis,
        screener: RiskScreener,
        batch_size: int = 100,
        batch_timeout_ms: int = 500
    ):
        self.redis = redis
        self.screener = screener
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.running = False

    async def start(self):
        """Start consuming events"""
        self.running = True

        # Create consumer group
        try:
            await self.redis.xgroup_create(
                settings.input_stream,
                settings.consumer_group,
                id="$",
                mkstream=True
            )
        except Exception as e:
            # Group might already exist
            logger.debug(f"Consumer group creation: {e}")

        # Start consumer
        asyncio.create_task(self._consume_loop())
        logger.info("StreamConsumer started")

    async def stop(self):
        """Stop consuming"""
        self.running = False
        logger.info("StreamConsumer stopped")

    async def _consume_loop(self):
        """Main consumer loop"""
        while self.running:
            try:
                # Read batch from stream
                messages = await self.redis.xreadgroup(
                    settings.consumer_group,
                    settings.consumer_name,
                    {settings.input_stream: ">"},
                    count=self.batch_size,
                    block=self.batch_timeout_ms
                )

                if not messages:
                    continue

                # Process batch
                await self._process_batch(messages)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Consumer error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Backoff on error

    async def _process_batch(self, messages: List[Tuple[str, List[Tuple[str, Dict]]]]):
        """Process batch of events"""

        stream_messages = messages[0][1] if messages else []

        for message_id, data in stream_messages:
            try:
                # Parse event from Redis data
                event = Event(
                    event_id=data.get(b"event_id", b"").decode(),
                    event_type=data.get(b"event_type", b"").decode(),
                    platform=data.get(b"platform", b"").decode(),
                    channel_id=data.get(b"channel_id", b"").decode(),
                    user_id=data.get(b"user_id", b"").decode(),
                    author_id=data.get(b"author_id", b"").decode(),
                    content=data.get(b"content", b"").decode(),
                    metadata=json.loads(data.get(b"metadata", b"{}")),
                    created_at=datetime.fromisoformat(
                        data.get(b"created_at", b"").decode()
                    )
                )

                # Screen event
                prediction = await self.screener.screen(event)

                # Route to output stream
                output_stream = (
                    settings.output_stream_high
                    if prediction.risk_score > settings.screening_threshold
                    else settings.output_stream_low
                )

                # Add to output stream
                output_data = prediction.dict()
                output_data["event_id"] = event.event_id

                # Flatten nested objects for Redis
                flat_output = {}
                for k, v in output_data.items():
                    if isinstance(v, (dict, list)):
                        flat_output[k] = json.dumps(v)
                    else:
                        flat_output[k] = str(v)

                await self.redis.xadd(output_stream, flat_output)

                # Acknowledge message
                await self.redis.xack(
                    settings.input_stream,
                    settings.consumer_group,
                    message_id
                )

                logger.debug(f"Processed {event.event_id} → {output_stream}")

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)


# ============ FastAPI App ============

redis_client: Optional[Redis] = None
embedding_model: Optional[EmbeddingModel] = None
vector_store: Optional[FAISSVectorStore] = None
risk_screener: Optional[RiskScreener] = None
stream_consumer: Optional[StreamConsumer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""

    global redis_client, embedding_model, vector_store, risk_screener, stream_consumer

    # Startup
    logger.info(f"Starting {settings.service_name}")

    # Initialize components
    redis_client = await aioredis.create_redis_pool(
        f"redis://{settings.redis_host}:{settings.redis_port}",
        password=settings.redis_password or None
    )

    embedding_model = EmbeddingModel()
    vector_store = FAISSVectorStore(embedding_model.embedding_dim)
    risk_screener = RiskScreener(embedding_model, vector_store, redis_client)
    stream_consumer = StreamConsumer(redis_client, risk_screener)

    await stream_consumer.start()

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.service_name}")
    await stream_consumer.stop()
    redis_client.close()
    await redis_client.wait_closed()


app = FastAPI(
    title="Risk Screening Service",
    description="Ultra-fast harassment risk classification",
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
        "device": settings.device,
        "embedding_model": settings.screening_model
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
            "model_loaded": embedding_model is not None
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


# ============ API Endpoints ============

@app.post("/v1/risk/screen")
async def screen_event(
    event_id: str,
    content: str,
    api_key: str = Header(None)
) -> Dict[str, Any]:
    """
    Screen individual event.

    Primarily for testing. Production uses Redis Streams.
    """

    if api_key not in settings.valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not risk_screener:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Create minimal event
    event = Event(
        event_id=event_id,
        event_type="message_created",
        platform="web_chat",
        channel_id="test",
        user_id="test_user",
        author_id="test_author",
        content=content,
        metadata={
            "timestamp": datetime.utcnow(),
            "user_reputation": 0.5,
            "author_reputation": 0.5
        }
    )

    prediction = await risk_screener.screen(event)

    return prediction.dict()


@app.get("/v1/metrics")
async def get_metrics():
    """Return service metrics"""
    try:
        if not redis_client:
            raise RuntimeError("Redis not initialized")

        queue_depth = await redis_client.xlen(settings.input_stream)

        return {
            "model": settings.screening_model,
            "device": settings.device,
            "queue_depth": queue_depth,
            "threshold": settings.screening_threshold,
            "embedding_dim": embedding_model.embedding_dim if embedding_model else 0
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=settings.log_level)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower()
    )
