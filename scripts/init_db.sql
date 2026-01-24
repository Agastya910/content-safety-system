CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    platform VARCHAR(50) NOT NULL,
    reputation FLOAT DEFAULT 0.0,
    violation_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(255) UNIQUE NOT NULL,
    platform VARCHAR(50) NOT NULL,
    content TEXT,
    risk_score FLOAT,
    risk_category VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    channel_id VARCHAR(255),
    escalation_score FLOAT DEFAULT 0.0,
    message_count INT DEFAULT 0,
    harassment_flags_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS actions (
    id SERIAL PRIMARY KEY,
    action_id VARCHAR(255) UNIQUE NOT NULL,
    event_id VARCHAR(255) REFERENCES events(event_id),
    action_type VARCHAR(50),
    user_id VARCHAR(255),
    applied_at TIMESTAMP,
    reason_code VARCHAR(100)
);

CREATE INDEX idx_events_risk ON events(risk_score DESC);
CREATE INDEX idx_sessions_escalation ON sessions(escalation_score DESC);
