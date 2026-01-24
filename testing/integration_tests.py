# testing/integration_tests.py
"""
Integration tests for Event Ingestion and Risk Screening services.

Run with: pytest -v testing/integration_tests.py
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any
import aioredis
import httpx
import pytest
from uuid import uuid4


# ============ Configuration ============

BASE_URL_INGESTION = "http://localhost:8001"
BASE_URL_SCREENING = "http://localhost:8002"
REDIS_URL = "redis://localhost:6379"
API_KEY = "test-key"


# ============ Fixtures ============

@pytest.fixture
async def redis_client():
    """Redis client fixture"""
    client = await aioredis.create_redis_pool(REDIS_URL)
    yield client
    client.close()
    await client.wait_closed()


@pytest.fixture
def http_client():
    """HTTP client fixture"""
    with httpx.Client(timeout=30.0) as client:
        yield client


# ============ Event Ingestion Tests ============

class TestEventIngestion:
    """Test Event Ingestion Service"""

    @pytest.mark.asyncio
    async def test_health_check(self, http_client):
        """Test health check endpoint"""
        response = http_client.get(f"{BASE_URL_INGESTION}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "event-ingestion"

    @pytest.mark.asyncio
    async def test_readiness_check(self, http_client):
        """Test readiness check endpoint"""
        response = http_client.get(f"{BASE_URL_INGESTION}/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["redis"] == "ok"

    def test_single_event_ingestion(self, http_client):
        """Test ingesting single event"""
        event_id = str(uuid4())

        payload = {
            "event_id": event_id,
            "event_type": "message_created",
            "platform": "discord",
            "channel_id": "ch-123",
            "user_id": "user-123",
            "author_id": "auth-123",
            "content": "This is a test message",
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "user_reputation": 0.8,
                "author_reputation": 0.6
            }
        }

        response = http_client.post(
            f"{BASE_URL_INGESTION}/v1/events/ingest",
            json=payload,
            headers={"api-key": API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["event_id"] == event_id
        assert data["status"] == "queued"
        assert "ingestion_time_ms" in data

    def test_batch_ingestion(self, http_client):
        """Test ingesting batch of events"""
        events = [
            {
                "event_id": str(uuid4()),
                "event_type": "message_created",
                "platform": "discord",
                "user_id": "user-123",
                "author_id": "auth-123",
                "content": f"Test message {i}",
                "metadata": {"user_reputation": 0.5}
            }
            for i in range(10)
        ]

        payload = {"events": events}

        response = http_client.post(
            f"{BASE_URL_INGESTION}/v1/events/ingest-batch",
            json=payload,
            headers={"api-key": API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 10
        assert data["queued"] == 10
        assert data["duplicates"] == 0
        assert data["errors"] == 0

    def test_validation_error_missing_field(self, http_client):
        """Test validation error for missing field"""
        payload = {
            "event_id": str(uuid4()),
            "event_type": "message_created",
            # Missing platform, user_id, author_id, content
        }

        response = http_client.post(
            f"{BASE_URL_INGESTION}/v1/events/ingest",
            json=payload,
            headers={"api-key": API_KEY}
        )

        assert response.status_code == 400
        assert "Missing required fields" in response.json()["detail"]

    def test_validation_error_content_too_short(self, http_client):
        """Test validation error for content too short"""
        payload = {
            "event_id": str(uuid4()),
            "event_type": "message_created",
            "platform": "discord",
            "user_id": "user-123",
            "author_id": "auth-123",
            "content": "",  # Empty content
            "metadata": {}
        }

        response = http_client.post(
            f"{BASE_URL_INGESTION}/v1/events/ingest",
            json=payload,
            headers={"api-key": API_KEY}
        )

        assert response.status_code == 400
        assert "Content too short" in response.json()["detail"]

    def test_invalid_api_key(self, http_client):
        """Test rejection of invalid API key"""
        payload = {
            "event_id": str(uuid4()),
            "event_type": "message_created",
            "platform": "discord",
            "user_id": "user-123",
            "author_id": "auth-123",
            "content": "Test message",
        }

        response = http_client.post(
            f"{BASE_URL_INGESTION}/v1/events/ingest",
            json=payload,
            headers={"api-key": "invalid-key"}
        )

        assert response.status_code == 401

    def test_duplicate_detection(self, http_client):
        """Test duplicate event detection"""
        event_id_1 = str(uuid4())

        # First event
        payload_1 = {
            "event_id": event_id_1,
            "event_type": "message_created",
            "platform": "discord",
            "user_id": "user-123",
            "author_id": "auth-123",
            "content": "Duplicate test message",
        }

        response_1 = http_client.post(
            f"{BASE_URL_INGESTION}/v1/events/ingest",
            json=payload_1,
            headers={"api-key": API_KEY}
        )

        assert response_1.status_code == 200
        assert response_1.json()["status"] == "queued"

        # Duplicate event (same content, author, platform)
        event_id_2 = str(uuid4())
        payload_2 = {
            **payload_1,
            "event_id": event_id_2,
        }

        response_2 = http_client.post(
            f"{BASE_URL_INGESTION}/v1/events/ingest",
            json=payload_2,
            headers={"api-key": API_KEY}
        )

        assert response_2.status_code == 200
        assert response_2.json()["status"] == "duplicate"

    def test_metrics_endpoint(self, http_client):
        """Test metrics endpoint"""
        response = http_client.get(f"{BASE_URL_INGESTION}/v1/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "queue_depth" in data
        assert "max_queue_depth" in data
        assert "queue_capacity_percent" in data


# ============ Risk Screening Tests ============

class TestRiskScreening:
    """Test Risk Screening Service"""

    def test_health_check(self, http_client):
        """Test health check endpoint"""
        response = http_client.get(f"{BASE_URL_SCREENING}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "risk-screening"

    def test_readiness_check(self, http_client):
        """Test readiness check endpoint"""
        response = http_client.get(f"{BASE_URL_SCREENING}/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_screen_clean_content(self, http_client):
        """Test screening clean content"""
        response = http_client.post(
            f"{BASE_URL_SCREENING}/v1/risk/screen",
            params={
                "event_id": str(uuid4()),
                "content": "This is a normal, friendly message"
            },
            headers={"api-key": API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["risk_score"] < 0.5
        assert data["risk_category"] in ["low_risk", "spam"]

    def test_screen_toxic_content(self, http_client):
        """Test screening toxic content"""
        response = http_client.post(
            f"{BASE_URL_SCREENING}/v1/risk/screen",
            params={
                "event_id": str(uuid4()),
                "content": "You are such a stupid idiot!!!"
            },
            headers={"api-key": API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["risk_score"] > 0.5
        assert data["risk_category"] in ["toxic", "targeted_harassment"]
        assert "offensive_language" in [f.lower() for f in data["flags"]]

    def test_screen_spam_content(self, http_client):
        """Test screening spam content"""
        response = http_client.post(
            f"{BASE_URL_SCREENING}/v1/risk/screen",
            params={
                "event_id": str(uuid4()),
                "content": "CLICK HERE!!! AMAZING OFFER!!! BUY NOW!!!"
            },
            headers={"api-key": API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["risk_score"] >= 0.5
        assert "spam" in data["risk_category"].lower() or "caps" in str(data["flags"]).lower()

    def test_metrics_endpoint(self, http_client):
        """Test metrics endpoint"""
        response = http_client.get(f"{BASE_URL_SCREENING}/v1/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "queue_depth" in data
        assert "threshold" in data
        assert "embedding_dim" in data


# ============ End-to-End Tests ============

class TestEndToEnd:
    """End-to-end integration tests"""

    def test_event_flow(self, http_client):
        """Test complete event flow: ingest -> screen -> route"""
        event_id = str(uuid4())

        # 1. Ingest event
        payload = {
            "event_id": event_id,
            "event_type": "message_created",
            "platform": "discord",
            "user_id": "user-123",
            "author_id": "auth-123",
            "content": "This content contains offensive language like terrible and bad",
        }

        response = http_client.post(
            f"{BASE_URL_INGESTION}/v1/events/ingest",
            json=payload,
            headers={"api-key": API_KEY}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "queued"

        # 2. Give screening service time to consume
        time.sleep(2)

        # 3. Screen event
        response = http_client.post(
            f"{BASE_URL_SCREENING}/v1/risk/screen",
            params={
                "event_id": event_id,
                "content": payload["content"]
            },
            headers={"api-key": API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "risk_category" in data
        assert "screening_time_ms" in data
        assert data["screening_time_ms"] < 100  # Should be < 100ms

    @pytest.mark.asyncio
    async def test_high_throughput(self, redis_client, http_client):
        """Test high throughput scenario"""
        num_events = 100
        start_time = time.time()

        # Ingest batch
        events = [
            {
                "event_id": str(uuid4()),
                "event_type": "message_created",
                "platform": "discord",
                "user_id": "user-123",
                "author_id": "auth-123",
                "content": f"Test message {i}",
            }
            for i in range(num_events)
        ]

        response = http_client.post(
            f"{BASE_URL_INGESTION}/v1/events/ingest-batch",
            json={"events": events},
            headers={"api-key": API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["queued"] == num_events

        elapsed = time.time() - start_time
        throughput = num_events / elapsed

        print(f"\nThroughput: {throughput:.0f} events/sec")
        assert throughput > 100  # At least 100 events/sec


# ============ Performance Tests ============

class TestPerformance:
    """Performance benchmarks"""

    def test_ingestion_latency(self, http_client):
        """Test ingestion latency (P99 < 100ms)"""
        latencies = []

        for i in range(100):
            payload = {
                "event_id": str(uuid4()),
                "event_type": "message_created",
                "platform": "discord",
                "user_id": "user-123",
                "author_id": "auth-123",
                "content": f"Performance test message {i}",
            }

            start = time.perf_counter()
            response = http_client.post(
                f"{BASE_URL_INGESTION}/v1/events/ingest",
                json=payload,
                headers={"api-key": API_KEY}
            )
            elapsed = (time.perf_counter() - start) * 1000

            assert response.status_code == 200
            latencies.append(elapsed)

        latencies.sort()
        p50 = latencies[50]
        p99 = latencies[99]

        print(f"\nIngestion Latency - P50: {p50:.2f}ms, P99: {p99:.2f}ms")
        assert p99 < 100  # P99 should be < 100ms

    def test_screening_latency(self, http_client):
        """Test screening latency (P99 < 50ms)"""
        latencies = []

        for i in range(50):
            start = time.perf_counter()
            response = http_client.post(
                f"{BASE_URL_SCREENING}/v1/risk/screen",
                params={
                    "event_id": str(uuid4()),
                    "content": f"Test message {i}"
                },
                headers={"api-key": API_KEY}
            )
            elapsed = (time.perf_counter() - start) * 1000

            assert response.status_code == 200
            latencies.append(elapsed)

        latencies.sort()
        p50 = latencies[25]
        p99 = latencies[49]

        print(f"\nScreening Latency - P50: {p50:.2f}ms, P99: {p99:.2f}ms")
        assert p99 < 200  # P99 should be < 200ms (includes network)


# ============ Load Test (Locust) ============

# testing/locustfile.py
"""
Load test configuration for Locust.

Run with: locust -f testing/locustfile.py --host=http://localhost:8001
"""

from locust import HttpUser, task, between
from uuid import uuid4


class EventIngestionUser(HttpUser):
    """Simulates event ingestion users"""

    wait_time = between(0.1, 0.5)

    def on_start(self):
        """Setup"""
        self.api_key = "test-key"

    @task(3)
    def ingest_single_event(self):
        """Ingest single event"""
        payload = {
            "event_id": str(uuid4()),
            "event_type": "message_created",
            "platform": "discord",
            "user_id": "user-123",
            "author_id": "auth-123",
            "content": f"Load test message {uuid4()}",
        }

        self.client.post(
            "/v1/events/ingest",
            json=payload,
            headers={"api-key": self.api_key}
        )

    @task(1)
    def ingest_batch(self):
        """Ingest batch of events"""
        events = [
            {
                "event_id": str(uuid4()),
                "event_type": "message_created",
                "platform": "discord",
                "user_id": "user-123",
                "author_id": "auth-123",
                "content": f"Batch message {i}",
            }
            for i in range(50)
        ]

        self.client.post(
            "/v1/events/ingest-batch",
            json={"events": events},
            headers={"api-key": self.api_key}
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
