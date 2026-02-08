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
import os
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
import redis.asyncio as redis
from redis.asyncio import Redis
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import opentelemetry.trace as trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from pathlib import Path

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
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "redis_dev_password")

        # Screening
        self.screening_model = os.getenv("SCREENING_MODEL", "all-MiniLM-L6-v2")
        self.screening_threshold = float(os.getenv("SCREENING_THRESHOLD", "0.7"))
        self.embedding_cache_ttl = int(os.getenv("EMBEDDING_CACHE_TTL", "604800"))  # 7 days

        # Model
        self.model_batch_size = int(os.getenv("MODEL_BATCH_SIZE", "32"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp32 = os.getenv("USE_FP32", "false").lower() == "true"

        # Qdrant
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.collection_name = "abuse_patterns"

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


# ============ Embedding Model (Ollama) ============

class EmbeddingModel:
    """
    Manages embeddings using Ollama's nomic-embed-text model.

    Benefits of Ollama:
    - Unified API for embeddings and LLM
    - No complex dependency management
    - Works with local and cloud models
    - 768-dim embeddings from nomic-embed-text
    """

    def __init__(self):
        self.model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.embedding_dim = 768  # nomic-embed-text dimension
        logger.info(f"Using Ollama embeddings: {self.model_name} at {self.ollama_host}")

    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed batch of texts using Ollama.

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        import httpx

        embeddings = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                response = await client.post(
                    f"{self.ollama_host}/api/embeddings",
                    json={"model": self.model_name, "prompt": text}
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])

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
        "you are", "you're", "your", "you all", "y'all"
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


# ============ Qdrant Vector Store ============

class QdrantVectorStore:
    """
    Qdrant-backed vector store for persistence and similarity search.
    """

    def __init__(self, embedding_dim: int = 768):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None
        )
        self.embedding_dim = embedding_dim
        self.ensure_collection()
        self.load_seed_data_if_empty()

    def ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            self.client.get_collection(settings.collection_name)
            logger.info(f"Connected to collection: {settings.collection_name}")
        except Exception:
            logger.info(f"Creating collection: {settings.collection_name}")
            self.client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.embedding_dim,
                    distance=qmodels.Distance.COSINE
                )
            )

    def load_seed_data_if_empty(self):
        """Load seed corpus if collection is empty"""
        try:
            count = self.client.count(settings.collection_name).count
            if count == 0:
                logger.info("Collection empty. Loading seed data...")
                seed_path = Path(__file__).parent / "data" / "abusive_patterns.json"

                if not seed_path.exists():
                    logger.warning(f"Seed file not found at {seed_path}")
                    return

                with open(seed_path, "r") as f:
                    patterns = json.load(f)

                if not patterns:
                    return

                # Generate embeddings for seed data
                # Note: We need the embedding model here.
                # Ideally, this should be async or done at startup differently.
                # For now, we'll assume this runs fast or use a separate loading mechanism.
                # However, since we are inside __init__ (sync), we can't await.
                # We will defer this to an async init method or handle it in lifespan.
                pass

        except Exception as e:
            logger.error(f"Failed to check/load seed data: {e}")

    async def seed(self, embedding_model: EmbeddingModel):
        """Async seeder called from lifespan"""
        try:
            count = self.client.count(settings.collection_name).count
            if count > 0:
                return

            seed_path = Path(__file__).parent / "data" / "abusive_patterns.json"
            if not seed_path.exists():
                return

            logger.info(f"Seeding from {seed_path}...")
            with open(seed_path, "r") as f:
                patterns = json.load(f)

            if not patterns:
                return

            texts = [p["text"] for p in patterns]
            embeddings = await embedding_model.embed(texts)

            points = []
            for i, (text, emb) in enumerate(zip(texts, embeddings)):
                points.append(qmodels.PointStruct(
                    id=i,  # Simple integer IDs for seed data
                    vector=emb.tolist(),
                    payload={
                        "text": text,
                        "type": "seed_abuse_pattern",
                        "category": patterns[i].get("category", "HARASSMENT"),
                        "severity": patterns[i].get("severity", "MEDIUM")
                    }
                ))

            self.client.upsert(
                collection_name=settings.collection_name,
                points=points
            )
            logger.info(f"Seeding complete: {len(points)} patterns loaded.")

        except Exception as e:
            logger.error(f"Seeding failed: {e}")

    async def search(self, embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for k most similar embeddings.
        Returns list of matched payloads with scores.
        """
        try:
            # Qdrant expects list[float]
            vector = embedding.tolist()

            results = self.client.search(
                collection_name=settings.collection_name,
                query_vector=vector,
                limit=k
            )

            matches = []
            for hit in results:
                matches.append({
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "category": hit.payload.get("category", "UNKNOWN")
                })
            return matches

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []


# ============ Risk Screener ============

class RiskScreener:
    """
    Main screening engine combining:
    - ML embeddings + similarity search
    - Behavioral heuristics
    - Batch processing
    """

    def __init__(self, embedding_model: EmbeddingModel, vector_store: QdrantVectorStore, redis: Redis):
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
                        # Redis returns bytes, decode to str for fromhex
                        if isinstance(cached_embedding, bytes):
                            cached_embedding = cached_embedding.decode('utf-8')
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
                            ex=settings.embedding_cache_ttl
                        )

                    span.set_attribute("cache_hit", False)

                # 3. Behavioral heuristics score
                heuristic_score, heuristic_flags = BehavioralHeuristics.score(event.content)

                # 4. Semantic Similarity Score
                similar_events = await self.vector_store.search(embedding, k=3)

                similarity_score = 0.0
                if similar_events:
                    # Top match score (Cosine similarity is -1 to 1, but Qdrant usually returns 0-1 for normalized)
                    # We map it to a risk probability
                    top_score = similar_events[0]['score']
                    # Heuristic mapping: 0.8+ similarity -> High risk
                    similarity_score = max(0.0, top_score)

                # 5. Hybrid Scoring Formula
                # combined_score = 0.4 * heuristic_score + 0.6 * similarity_score
                # This ensures meaningful contribution from both signals.
                combined_score = (0.4 * heuristic_score) + (0.6 * similarity_score)
                combined_score = min(combined_score, 1.0) # Cap at 1.0

                # 6. Classify risk
                if combined_score > 0.85:
                    risk_category = RiskCategory.TARGETED_HARASSMENT
                elif combined_score > 0.70:
                    risk_category = RiskCategory.TOXIC
                elif combined_score > 0.50:
                    risk_category = RiskCategory.SPAM
                else:
                    risk_category = RiskCategory.LOW_RISK

                # 7. Calculate latency

                # Calculate latency
                screening_time_ms = (time.perf_counter() - start_time) * 1000

                # Create prediction
                prediction = RiskPrediction(
                    event_id=event.event_id,
                    risk_score=combined_score,
                    risk_category=risk_category,
                    confidence=max(heuristic_score, similarity_score),
                    flags=[str(f) for f in heuristic_flags],
                    details={
                        "heuristic_score": float(heuristic_score),
                        "embedding_score": float(similarity_score),
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
vector_store: Optional[QdrantVectorStore] = None
risk_screener: Optional[RiskScreener] = None
stream_consumer: Optional[StreamConsumer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""

    global redis_client, embedding_model, vector_store, risk_screener, stream_consumer

    # Startup
    logger.info(f"Starting {settings.service_name}")

    # Initialize components
    redis_client = redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}",
        password=settings.redis_password or None,
        decode_responses=False  # Important for streams to handle binary data if needed, or True if purely text
    )

    embedding_model = EmbeddingModel()
    vector_store = QdrantVectorStore(embedding_model.embedding_dim)

    # Run seeder
    await vector_store.seed(embedding_model)

    risk_screener = RiskScreener(embedding_model, vector_store, redis_client)
    stream_consumer = StreamConsumer(redis_client, risk_screener)

    await stream_consumer.start()

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.service_name}")
    await stream_consumer.stop()
    await redis_client.aclose()


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
