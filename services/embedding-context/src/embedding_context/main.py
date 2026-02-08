import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import httpx
import numpy as np

# Shared models
from safety_system.core.models import (
    ContextData, SimilarEvent, TemporalFeatures, RiskPrediction, Event
)

# Configuration
class Settings:
    def __init__(self):
        self.service_name = "embedding-context"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

settings = Settings()

logger = logging.getLogger("embedding_context")
logging.basicConfig(level=settings.log_level)

# Collections (768-dim for nomic-embed-text)
COLLECTIONS = {
    "abuse_patterns": 768,
    "actor_profiles": 768,
    "campaign_clusters": 768,
    "victim_profiles": 768
}

# Service State
class ServiceState:
    def __init__(self):
        self.qdrant: Optional[QdrantClient] = None

state = ServiceState()

app = FastAPI(title="Embedding Context Service")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Embedding Context Service...")

    # 1. Init Qdrant
    state.qdrant = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key if settings.qdrant_api_key else None
    )

    # 2. Log Ollama config (no init needed - stateless HTTP calls)
    logger.info(f"Using Ollama embeddings: {settings.ollama_embed_model} at {settings.ollama_host}")

    # 3. Ensure Collections
    for name, dim in COLLECTIONS.items():
        try:
            state.qdrant.get_collection(name)
            logger.info(f"Collection exists: {name}")
        except Exception:
            logger.info(f"Creating collection: {name}")
            state.qdrant.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
            )

@app.get("/health")
async def health():
    return {"status": "ok", "service": "embedding-context"}

@app.get("/ready")
async def ready():
    if not state.qdrant or not state.embedder:
         raise HTTPException(status_code=503, detail="Service not initialized")
    return {"status": "ready"}

class AssembleRequest(BaseModel):
    event: Dict[str, Any] # Raw event dict or Event model
    risk_prediction: Optional[Dict[str, Any]] = None

@app.post("/v1/context/assemble", response_model=ContextData)
async def assemble_context(request: AssembleRequest):
    """
    Assemble rich context for an event by querying all vector collections.
    """
    if not state.qdrant or not state.embedder:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Parse event
        event_data = request.event
        content = event_data.get("content", "")
        event_id = event_data.get("event_id", "unknown")
        user_id = event_data.get("user_id", "unknown") # Victim
        author_id = event_data.get("author_id", "unknown") # Actor

        if not content:
             raise HTTPException(status_code=400, detail="Event content missing")

        # 1. Generate Embedding
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, state.embedder.encode, content)
        embedding_list = embedding.tolist()

        # 2. Parallel Search
        # We search specific collections for different logical purposes

        # A. Abuse Patterns (Similar past cases)
        abuse_search = state.qdrant.search(
            collection_name="abuse_patterns",
            query_vector=embedding_list,
            limit=5
        )

        # B. Actor Profiles (Has this actor done this before?)
        # For actor search, ideally we assume we have an embedding for the actor's *behavioral history*.
        # For now, we search the actor_profiles collection using the *current event* embedding
        # as a proxy for "is this event similar to known bad actor behaviors".
        # In a real system, we'd look up the actor_id directly.
        # We will filter by author_id if possible, but for now just semantic search.
        actor_search = state.qdrant.search(
            collection_name="actor_profiles",
            query_vector=embedding_list,
            limit=3
        )

        # C. Campaign Clusters (Is this part of a raid?)
        campaign_search = state.qdrant.search(
            collection_name="campaign_clusters",
            query_vector=embedding_list,
            limit=1
        )

        # D. Victim Profiles (Is this user usually targeted?)
        victim_search = state.qdrant.search(
            collection_name="victim_profiles",
            query_vector=embedding_list,
            limit=3
        )

        # 3. Assemble Results
        def to_similar_events(hits):
            results = []
            for hit in hits:
                payload = hit.payload or {}
                results.append(SimilarEvent(
                    event_id=payload.get("event_id", "unknown"),
                    similarity=hit.score,
                    content=payload.get("text", "") or payload.get("content", ""),
                    timestamp=datetime.utcnow(), # Placeholder
                    author_id=payload.get("author_id", "unknown")
                ))
            return results

        # 4. Construct ContextData
        context_data = ContextData(
            event_id=event_id,
            embedding=embedding_list,
            similar_events=to_similar_events(abuse_search),
            actor_context=to_similar_events(actor_search),
            campaign_context=to_similar_events(campaign_search),
            victim_context=to_similar_events(victim_search),
            session_temporal_features=TemporalFeatures() # Stub for now
        )

        return context_data

    except Exception as e:
        logger.error(f"Context assembly failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
