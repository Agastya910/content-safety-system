# services/action-executor/src/action_executor/main.py
"""
Action Executor Service - Enforcement Actions

Takes reasoning output and executes appropriate enforcement actions:
- shadow_hide: Hide content from others, author still sees it
- blur: Blur content with warning overlay
- collapse_thread: Collapse reply threads
- throttle: Rate limit the author
- warn_user: Send warning to author
- escalate_human: Flag for human review

Actions are logged and reversible.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

# Configuration
class Settings:
    def __init__(self):
        self.service_name = "action-executor"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("action_executor")

# ============ Models ============

class ActionType(str, Enum):
    NONE = "none"
    SHADOW_HIDE = "shadow_hide"
    BLUR = "blur"
    COLLAPSE_THREAD = "collapse_thread"
    THROTTLE = "throttle"
    WARN_USER = "warn_user"
    ESCALATE_HUMAN = "escalate_human"

class ActionRequest(BaseModel):
    event_id: str
    content_id: str
    author_id: Optional[str] = None
    harm_score: float = Field(ge=0.0, le=1.0)
    recommended_actions: List[ActionType]
    explanation: str = ""

class ActionResult(BaseModel):
    action: ActionType
    success: bool
    message: str

class ActionResponse(BaseModel):
    event_id: str
    content_id: str
    executed_actions: List[ActionResult]
    total_actions: int
    execution_time_ms: int

# ============ Action Handlers ============

async def execute_shadow_hide(content_id: str, author_id: str) -> ActionResult:
    """Hide content from everyone except author."""
    logger.info(f"SHADOW_HIDE: content={content_id}, author={author_id}")
    # In production: Update content visibility in database
    return ActionResult(
        action=ActionType.SHADOW_HIDE,
        success=True,
        message=f"Content {content_id} shadow-hidden"
    )

async def execute_blur(content_id: str) -> ActionResult:
    """Apply blur overlay with warning."""
    logger.info(f"BLUR: content={content_id}")
    # In production: Set blur flag on content
    return ActionResult(
        action=ActionType.BLUR,
        success=True,
        message=f"Content {content_id} blurred with warning"
    )

async def execute_collapse_thread(content_id: str) -> ActionResult:
    """Collapse thread starting from this content."""
    logger.info(f"COLLAPSE_THREAD: content={content_id}")
    return ActionResult(
        action=ActionType.COLLAPSE_THREAD,
        success=True,
        message=f"Thread from {content_id} collapsed"
    )

async def execute_throttle(author_id: str) -> ActionResult:
    """Rate limit the author temporarily."""
    if not author_id:
        return ActionResult(
            action=ActionType.THROTTLE,
            success=False,
            message="No author_id provided"
        )
    logger.info(f"THROTTLE: author={author_id}")
    # In production: Add to rate limit cache
    return ActionResult(
        action=ActionType.THROTTLE,
        success=True,
        message=f"Author {author_id} throttled for 10 minutes"
    )

async def execute_warn_user(author_id: str, explanation: str) -> ActionResult:
    """Send warning notification to author."""
    if not author_id:
        return ActionResult(
            action=ActionType.WARN_USER,
            success=False,
            message="No author_id provided"
        )
    logger.info(f"WARN_USER: author={author_id}")
    # In production: Send notification via platform API
    return ActionResult(
        action=ActionType.WARN_USER,
        success=True,
        message=f"Warning sent to {author_id}"
    )

async def execute_escalate_human(event_id: str, explanation: str) -> ActionResult:
    """Flag content for human moderator review."""
    logger.info(f"ESCALATE_HUMAN: event={event_id}")
    # In production: Add to moderation queue
    return ActionResult(
        action=ActionType.ESCALATE_HUMAN,
        success=True,
        message=f"Event {event_id} escalated for human review"
    )

# ============ App ============

app = FastAPI(
    title="Action Executor Service",
    description="Executes enforcement actions based on reasoning output",
    version="1.0.0"
)

@app.get("/health")
async def health():
    return {"status": "ok", "service": settings.service_name}

@app.get("/ready")
async def ready():
    return {"status": "ready"}

@app.post("/v1/execute", response_model=ActionResponse)
async def execute_actions(request: ActionRequest = Body(...)):
    """
    Execute enforcement actions based on harm score and recommendations.

    Returns list of executed actions with success/failure status.
    """
    import time

    start_time = time.perf_counter()
    executed_actions = []

    for action in request.recommended_actions:
        if action == ActionType.NONE:
            continue

        try:
            if action == ActionType.SHADOW_HIDE:
                result = await execute_shadow_hide(
                    request.content_id,
                    request.author_id or "unknown"
                )
            elif action == ActionType.BLUR:
                result = await execute_blur(request.content_id)
            elif action == ActionType.COLLAPSE_THREAD:
                result = await execute_collapse_thread(request.content_id)
            elif action == ActionType.THROTTLE:
                result = await execute_throttle(request.author_id)
            elif action == ActionType.WARN_USER:
                result = await execute_warn_user(
                    request.author_id,
                    request.explanation
                )
            elif action == ActionType.ESCALATE_HUMAN:
                result = await execute_escalate_human(
                    request.event_id,
                    request.explanation
                )
            else:
                result = ActionResult(
                    action=action,
                    success=False,
                    message=f"Unknown action: {action}"
                )

            executed_actions.append(result)

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            executed_actions.append(ActionResult(
                action=action,
                success=False,
                message=str(e)
            ))

    execution_time_ms = int((time.perf_counter() - start_time) * 1000)

    return ActionResponse(
        event_id=request.event_id,
        content_id=request.content_id,
        executed_actions=executed_actions,
        total_actions=len(executed_actions),
        execution_time_ms=execution_time_ms
    )

@app.get("/v1/actions")
async def list_action_types():
    """List all available action types."""
    return {
        "actions": [a.value for a in ActionType],
        "descriptions": {
            "none": "No action taken",
            "shadow_hide": "Hide content from others, author sees it",
            "blur": "Apply blur with warning overlay",
            "collapse_thread": "Collapse reply thread",
            "throttle": "Rate limit author temporarily",
            "warn_user": "Send warning to author",
            "escalate_human": "Flag for human review"
        }
    }
