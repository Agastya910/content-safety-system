from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
from datetime import datetime

app = FastAPI(title="Event Ingestion Service")

# Simple event model
class EventIngest(BaseModel):
    event_type: str
    platform: str
    channel_id: str
    user_id: str
    author_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

@app.get("/health")
async def health():
    return {"status": "ok", "service": "event-ingestion"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}

@app.post("/v1/events/ingest")
async def ingest_event(
    event: EventIngest,
    x_api_key: Optional[str] = Header(None)
):
    # Basic API key check
    if x_api_key not in ["test-key", "dev-key"]:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Generate event ID
    event_id = f"evt_{uuid.uuid4().hex[:12]}"
    
    # TODO: In Prompt #2, this will push to Redis stream
    # For now, just acknowledge receipt
    
    return {
        "event_id": event_id,
        "status": "queued",
        "processing_url": f"/v1/events/{event_id}/status"
    }

@app.get("/v1/events/{event_id}/status")
async def event_status(event_id: str):
    # Placeholder
    return {
        "event_id": event_id,
        "status": "processing",
        "message": "Event received and queued for processing"
    }
