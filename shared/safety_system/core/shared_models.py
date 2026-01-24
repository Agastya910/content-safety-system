# shared/safety_system/core/models.py
"""
Shared data models for the safety system.

Used across all services for type consistency and validation.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


# ============ Enums ============

class Platform(str, Enum):
    """Supported platforms"""
    DISCORD = "discord"
    SLACK = "slack"
    WEB_CHAT = "web_chat"
    TELEGRAM = "telegram"
    TWITTER = "twitter"
    REDDIT = "reddit"


class EventType(str, Enum):
    """Event types"""
    MESSAGE_CREATED = "message_created"
    MESSAGE_EDITED = "message_edited"
    MESSAGE_DELETED = "message_deleted"
    REACTION_ADDED = "reaction_added"
    REACTION_REMOVED = "reaction_removed"
    USER_MENTIONED = "user_mentioned"
    THREAD_CREATED = "thread_created"
    CHANNEL_CREATED = "channel_created"


class RiskCategory(str, Enum):
    """Risk classification categories"""
    LOW_RISK = "low_risk"
    SPAM = "spam"
    TOXIC = "toxic"
    TARGETED_HARASSMENT = "targeted_harassment"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"


class RiskFlag(str, Enum):
    """Risk flags for detailed categorization"""
    ALL_CAPS_SPAM = "all_caps_spam"
    REPEATED_PUNCTUATION = "repeated_punctuation"
    OFFENSIVE_LANGUAGE = "offensive_language"
    TARGETING_BEHAVIOR = "targeting_behavior"
    PERSONAL_ATTACK = "personal_attack"
    IDENTITY_ATTACK = "identity_attack"
    SEXUAL_CONTENT = "sexual_content"
    VIOLENT_THREATS = "violent_threats"
    COORDINATED_ATTACK = "coordinated_attack"


class ActionType(str, Enum):
    """Actions taken on content"""
    DELETE = "delete"
    HIDE = "hide"
    QUARANTINE = "quarantine"
    SHADOW_BAN = "shadow_ban"
    FULL_BAN = "full_ban"
    ESCALATE = "escalate"
    MONITOR = "monitor"


# ============ Core Models ============

class EventMetadata(BaseModel):
    """Event metadata"""
    user_reputation: Optional[float] = Field(default=0.5, ge=0, le=1)
    author_reputation: Optional[float] = Field(default=0.5, ge=0, le=1)
    account_age_days: Optional[int] = Field(default=0, ge=0)
    previous_violations: Optional[int] = Field(default=0, ge=0)
    community_votes: Optional[int] = Field(default=0)

    class Config:
        extra = "allow"  # Allow additional fields


class Event(BaseModel):
    """Core event model"""
    event_id: str = Field(..., min_length=1, max_length=500)
    event_type: str
    platform: str
    channel_id: str = ""
    user_id: str = ""
    author_id: str = ""
    content: str = Field(..., min_length=1, max_length=10000)
    metadata: EventMetadata = Field(default_factory=EventMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('event_type', pre=True)
    def validate_event_type(cls, v):
        if isinstance(v, EventType):
            return v.value
        return v.lower()

    @validator('platform', pre=True)
    def validate_platform(cls, v):
        if isinstance(v, Platform):
            return v.value
        return v.lower()


class RiskPrediction(BaseModel):
    """Risk prediction from screening service"""
    event_id: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_category: RiskCategory
    confidence: float = Field(..., ge=0, le=1)
    flags: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    screening_time_ms: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TemporalFeatures(BaseModel):
    """Temporal features for behavioral analysis"""
    time_since_last_message_sec: Optional[int] = None
    messages_in_last_hour: int = 0
    messages_in_last_day: int = 0
    messages_in_last_week: int = 0
    avg_message_frequency_per_hour: float = 0.0
    # Escalation patterns
    toxic_messages_in_last_hour: int = 0
    escalation_trend: float = 0.0  # -1 to 1


class SimilarEvent(BaseModel):
    """Similar past event for context"""
    event_id: str
    similarity_score: float
    risk_category: RiskCategory
    created_at: datetime


class ContextData(BaseModel):
    """Context for reasoning service"""
    event: Event
    risk_prediction: RiskPrediction
    similar_events: List[SimilarEvent] = Field(default_factory=list)
    temporal_features: TemporalFeatures = Field(default_factory=TemporalFeatures)


class ReasoningResult(BaseModel):
    """Result from reasoning service"""
    event_id: str
    risk_category: RiskCategory
    confidence: float
    reasoning: str
    recommended_action: ActionType
    severity: int = Field(..., ge=1, le=10)
    escalate_to_human: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Action(BaseModel):
    """Action to take on content"""
    event_id: str
    action_type: ActionType
    reason: str
    duration_seconds: Optional[int] = None  # For temporary actions
    notify_user: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ActionResult(BaseModel):
    """Result of action execution"""
    action_id: str
    event_id: str
    action_type: ActionType
    success: bool
    message: str
    executed_at: datetime = Field(default_factory=datetime.utcnow)


class UserHistory(BaseModel):
    """User history and reputation"""
    user_id: str
    platform: str
    total_messages: int = 0
    violations: int = 0
    actions_taken: int = 0
    reputation_score: float = Field(default=0.5, ge=0, le=1)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)


# ============ API Models ============

class IngestionRequest(BaseModel):
    """Request to ingest event"""
    event_id: str
    event_type: str
    platform: str
    channel_id: str = ""
    user_id: str
    author_id: str
    content: str
    created_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IngestionResponse(BaseModel):
    """Response from ingestion"""
    event_id: str
    status: str  # "queued", "duplicate", "error"
    position: Optional[str] = None
    ingestion_time_ms: float


class RiskScreeningResponse(BaseModel):
    """Response from risk screening"""
    event_id: str
    risk_score: float
    risk_category: str
    confidence: float
    flags: List[str]
    screening_time_ms: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsResponse(BaseModel):
    """Service metrics"""
    service: str
    queue_depth: int
    throughput_events_per_sec: float
    avg_latency_ms: float
    error_rate: float
    uptime_seconds: int
