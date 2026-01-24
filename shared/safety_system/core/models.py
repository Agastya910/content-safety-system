# shared/safety_system/core/models.py
"""
Core data models for the Safety System.
Used across all microservices.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID, uuid4
import json

from pydantic import BaseModel, Field, validator


# ============ Enums ============

class EventType(str, Enum):
    MESSAGE_CREATED = "message_created"
    MESSAGE_EDITED = "message_edited"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"


class Platform(str, Enum):
    DISCORD = "discord"
    TWITCH = "twitch"
    WEB_CHAT = "web_chat"
    TELEGRAM = "telegram"


class RiskCategory(str, Enum):
    TARGETED_HARASSMENT = "targeted_harassment"
    SPAM = "spam"
    TOXIC = "toxic"
    HATE_SPEECH = "hate_speech"
    SELF_HARM = "self_harm"
    SEXUAL_CONTENT = "sexual_content"
    LOW_RISK = "low_risk"


class ActionType(str, Enum):
    WARNING = "warning"
    TIMEOUT = "timeout"
    MUTE = "mute"
    BAN = "ban"
    NONE = "none"


class ActionStatus(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    REVERSED = "reversed"
    EXPIRED = "expired"
    FAILED = "failed"


# ============ Events ============

class EventMetadata(BaseModel):
    """Metadata attached to events for context"""
    timestamp: datetime
    user_reputation: float = Field(default=0.0, ge=0.0, le=1.0)
    author_reputation: float = Field(default=0.0, ge=0.0, le=1.0)
    channel_history_count: int = 0
    is_reply_to: Optional[str] = None
    user_verified: bool = False
    extra: Dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    """Base event model"""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    platform: Platform
    channel_id: str
    user_id: str  # recipient/target
    author_id: str  # sender
    content: str
    metadata: EventMetadata
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


# ============ Risk Assessment ============

class RiskFlag(str, Enum):
    """Feature flags indicating risk signals"""
    REPEATED_NAME_CALLING = "repeated_name_calling"
    ALL_CAPS_SPAM = "ALL_CAPS_spam"
    REPEATED_PUNCTUATION = "repeated_punctuation"
    RAPID_MESSAGES = "rapid_messages"
    OFFENSIVE_LANGUAGE = "offensive_language"
    DOXXING_ATTEMPT = "doxxing_attempt"
    TARGETING_BEHAVIOR = "targeting_behavior"


class RiskPrediction(BaseModel):
    """ML model risk prediction"""
    event_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_category: RiskCategory
    confidence: float = Field(ge=0.0, le=1.0)
    flags: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    model_version: str = "v1.0"
    screening_time_ms: int = 0


# ============ Session & Temporal ============

class TemporalFeatures(BaseModel):
    """Features extracted from session temporal analysis"""
    message_count: int = 0
    time_span_minutes: float = 0.0
    avg_time_delta_seconds: float = 0.0
    escalation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    burst_detected: bool = False
    unique_targets: int = 0
    intervals: List[float] = Field(default_factory=list)  # Time deltas
    risk_scores: List[float] = Field(default_factory=list)  # Per-message risk


class Session(BaseModel):
    """Conversation session"""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    platform: Platform
    channel_id: str
    participants: List[str] = Field(default_factory=list)
    message_count: int = 0
    harassment_flags_count: int = 0
    temporal_features: TemporalFeatures = Field(default_factory=TemporalFeatures)
    status: str = "active"  # active, closed
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None


# ============ RAG & Context ============

class SimilarEvent(BaseModel):
    """Similar event retrieved from vector DB"""
    event_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    content: str
    timestamp: datetime
    author_id: str
    action_taken: Optional[str] = None
    action_outcome: Optional[str] = None


class ContextData(BaseModel):
    """Context retrieved for reasoning"""
    event_id: str
    embedding: List[float]
    similar_events: List[SimilarEvent] = Field(default_factory=list)
    session_temporal_features: TemporalFeatures = Field(default_factory=TemporalFeatures)


# ============ Reasoning ============

class EvidenceTrace(BaseModel):
    """Evidence supporting a reasoning decision"""
    type: str  # "similar_event", "temporal_escalation", "user_history"
    reference: str  # Event ID or description
    weight: float = Field(ge=0.0, le=1.0)
    explanation: Optional[str] = None


class ReasoningResult(BaseModel):
    """LLM reasoning output"""
    event_id: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_action: ActionType
    action_confidence: float = Field(ge=0.0, le=1.0)
    evidence_chain: List[EvidenceTrace] = Field(default_factory=list)
    fallback_rule_applied: bool = False
    reasoning_time_ms: int = 0


# ============ Actions ============

class ActionPayload(BaseModel):
    """Parameters for action execution"""
    duration_hours: Optional[int] = None
    reason: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """Action to be taken"""
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    event_id: str
    action_type: ActionType
    user_id: str  # Target user
    platform: Platform
    target_id: str  # Channel, DM, etc.
    reason_code: RiskCategory
    payload: ActionPayload = Field(default_factory=ActionPayload)
    status: ActionStatus = ActionStatus.PENDING
    applied_at: Optional[datetime] = None
    reversed_at: Optional[datetime] = None


class ActionResult(BaseModel):
    """Result of action execution"""
    action_id: str
    event_id: str
    status: ActionStatus
    applied_at: datetime
    platform_response: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ============ User Context ============

class UserHistory(BaseModel):
    """User violation history"""
    user_id: str
    prior_violations: int = 0
    warnings_issued: int = 0
    timeout_history: int = 0
    bans: int = 0
    days_since_last_violation: Optional[int] = None
    reputation: float = Field(default=0.0, ge=0.0, le=1.0)


# ============ API Requests/Responses ============

class EventIngestionRequest(BaseModel):
    """Webhook payload from platforms"""
    event_type: EventType
    platform: Platform
    channel_id: str
    user_id: str
    author_id: str
    content: str
    metadata: EventMetadata


class ScreeningRequest(BaseModel):
    """Internal request to risk screening service"""
    event_id: str
    content: str
    user_context: Optional[UserHistory] = None


class RetrieveContextRequest(BaseModel):
    """Request for RAG context retrieval"""
    event_id: str
    content: str
    user_id: str
    session_id: str
    top_k: int = 5


class ReasoningRequest(BaseModel):
    """Request for LLM reasoning"""
    event_id: str
    content: str
    risk_score: float
    similar_events: List[SimilarEvent] = Field(default_factory=list)
    session_features: TemporalFeatures = Field(default_factory=TemporalFeatures)
    user_history: Optional[UserHistory] = None


class ExecuteActionRequest(BaseModel):
    """Request to execute action"""
    event_id: str
    action_type: ActionType
    user_id: str
    platform: Platform
    target_id: str
    reason_code: RiskCategory
    payload: ActionPayload = Field(default_factory=ActionPayload)


# ============ Analytics ============

class DailyAnalytics(BaseModel):
    """Daily aggregated metrics"""
    date: str
    events_total: int
    events_screened: int
    harassment_detected: int
    actions_warning: int
    actions_timeout: int
    actions_mute: int
    actions_ban: int
    avg_screening_time_ms: float
    false_positive_rate: float
