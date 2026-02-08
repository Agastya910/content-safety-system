# services/reasoning/src/reasoning/main.py
"""
Reasoning Service - LLM-powered Harassment Analysis

Uses Ollama (qwen2:7b-instruct) to:
- Analyze harassment context and intent
- Generate nuanced harm scores
- Recommend enforcement actions
- Provide human-readable explanations

The LLM reasons about WHY content is harmful, not just IF it matches patterns.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import httpx

# Configuration
class Settings:
    def __init__(self):
        self.service_name = "reasoning"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2:7b-instruct-q4_0")
        self.timeout = int(os.getenv("LLM_TIMEOUT", "60"))

settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("reasoning")

# ============ Models ============

class ActionType(str, Enum):
    NONE = "none"
    SHADOW_HIDE = "shadow_hide"
    BLUR = "blur"
    COLLAPSE_THREAD = "collapse_thread"
    THROTTLE = "throttle"
    WARN_USER = "warn_user"
    ESCALATE_HUMAN = "escalate_human"

class ReasoningRequest(BaseModel):
    event_id: str
    content: str
    author_id: Optional[str] = None
    target_id: Optional[str] = None
    risk_score: float = 0.0
    similar_events: List[Dict[str, Any]] = []
    flags: List[str] = []

class ReasoningResponse(BaseModel):
    event_id: str
    harm_score: float = Field(ge=0.0, le=1.0)
    explanation: str
    recommended_actions: List[ActionType]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_time_ms: int

# ============ Prompt Template ============

REASONING_PROMPT = """You are a content safety AI analyzing potential harassment.

CONTENT TO ANALYZE:
"{content}"

CONTEXT:
- Initial Risk Score: {risk_score}
- Detected Flags: {flags}
- Similar Past Events: {similar_count} found

TASK:
Analyze this content for harassment, threats, or harmful intent.

OUTPUT FORMAT (JSON only, no markdown):
{{
  "harm_score": <0.0-1.0 float>,
  "explanation": "<1-2 sentence explanation>",
  "actions": ["none" | "shadow_hide" | "blur" | "collapse_thread" | "throttle" | "warn_user" | "escalate_human"],
  "confidence": <0.0-1.0 float>
}}

GUIDELINES:
- harm_score 0.0-0.3: Low risk, minor toxicity at most
- harm_score 0.3-0.6: Moderate, may need soft intervention
- harm_score 0.6-0.8: High, content should be hidden/blurred
- harm_score 0.8-1.0: Severe, immediate action + human review

Respond with ONLY the JSON object, no explanation outside."""

# ============ App ============

app = FastAPI(
    title="Reasoning Service",
    description="LLM-powered harassment reasoning and action recommendation",
    version="1.0.0"
)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": settings.service_name,
        "model": settings.ollama_model
    }

@app.get("/ready")
async def ready():
    # Check Ollama connectivity
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_host}/api/tags")
            resp.raise_for_status()
        return {"status": "ready", "ollama": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not ready: {e}")

@app.post("/v1/reason", response_model=ReasoningResponse)
async def reason(request: ReasoningRequest = Body(...)):
    """
    Analyze content and recommend enforcement actions.

    Uses LLM to reason about harassment context and intent.
    """
    import time
    import json

    start_time = time.perf_counter()

    # Build prompt
    prompt = REASONING_PROMPT.format(
        content=request.content[:500],  # Truncate for safety
        risk_score=request.risk_score,
        flags=", ".join(request.flags) if request.flags else "none",
        similar_count=len(request.similar_events)
    )

    try:
        # Call Ollama
        async with httpx.AsyncClient(timeout=settings.timeout) as client:
            response = await client.post(
                f"{settings.ollama_host}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 256
                    }
                }
            )
            response.raise_for_status()
            data = response.json()

        # Parse LLM response
        llm_output = data.get("response", "")
        logger.debug(f"LLM output: {llm_output}")

        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_start = llm_output.find("{")
            json_end = llm_output.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(llm_output[json_start:json_end])
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback based on risk score
            result = {
                "harm_score": min(request.risk_score + 0.1, 1.0),
                "explanation": "Analysis based on initial screening.",
                "actions": ["shadow_hide"] if request.risk_score > 0.7 else ["none"],
                "confidence": 0.5
            }

        # Map actions to enum
        # Map actions to enum with normalization
        actions = []
        raw_actions = result.get("actions", [])
        if isinstance(raw_actions, str):
            raw_actions = [raw_actions]

        for action in raw_actions:
            clean_action = str(action).lower().strip().replace(" ", "_")
            try:
                actions.append(ActionType(clean_action))
            except ValueError:
                logger.warning(f"Unknown action received: {action}")
                # Try to map common variations
                if "shadow" in clean_action:
                    actions.append(ActionType.SHADOW_HIDE)
                elif "warn" in clean_action:
                    actions.append(ActionType.WARN_USER)
                elif "ban" in clean_action or "block" in clean_action:
                    actions.append(ActionType.SHADOW_HIDE)

        # Safety net: If high harm but no valid actions, force escalation
        if result.get("harm_score", 0.0) >= 0.8 and not actions:
            logger.info("High harm score detected with no valid actions - forcing escalation")
            actions = [ActionType.SHADOW_HIDE, ActionType.ESCALATE_HUMAN]

        if not actions:
            actions = [ActionType.NONE]

        reasoning_time_ms = int((time.perf_counter() - start_time) * 1000)

        return ReasoningResponse(
            event_id=request.event_id,
            harm_score=float(result.get("harm_score", request.risk_score)),
            explanation=result.get("explanation", "No explanation provided."),
            recommended_actions=actions,
            confidence=float(result.get("confidence", 0.5)),
            reasoning_time_ms=reasoning_time_ms
        )

    except httpx.TimeoutException:
        logger.error(f"Ollama timeout for event {request.event_id}")
        raise HTTPException(status_code=504, detail="LLM reasoning timeout")
    except Exception as e:
        logger.error(f"Reasoning error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available Ollama models."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{settings.ollama_host}/api/tags")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
