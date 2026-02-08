import asyncio
import httpx
import json
import time
from datetime import datetime

# Service URLs
INGESTION_URL = "http://localhost:8001"
SCREENING_URL = "http://localhost:8002"
RETRIEVAL_URL = "http://localhost:8003"
REASONING_URL = "http://localhost:8004"
EXECUTOR_URL = "http://localhost:8005"

# Sample Harassment Event
EVENT = {
    "event_id": f"test-e2e-{int(time.time())}",
    "content": "I am going to kill you and your family. I know where you live.",
    "author_id": "bad-actor-001",
    "target_id": "victim-001",
    "timestamp": datetime.utcnow().isoformat() + "Z"
}

API_KEY = "test-key"

async def run_pipeline():
    print(f"🚀 Starting End-to-End Pipeline Test for event: {EVENT['event_id']}")

    headers = {"api-key": API_KEY}

    async with httpx.AsyncClient(timeout=120.0) as client:
        # 1. Ingestion (Simulated by calling Screening directly for now as Ingestion is async/stream)
        print("\n1️⃣  Risk Screening...")
        start = time.perf_counter()
        resp = await client.post(
            f"{SCREENING_URL}/v1/risk/screen",
            params={
                "event_id": EVENT["event_id"],
                "content": EVENT["content"]
            },
            headers=headers
        )
        print(f"   Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"❌ Screening failed: {resp.text}")
            return

        screening_result = resp.json()
        print(f"   Risk Score: {screening_result['risk_score']}")
        print(f"   Category: {screening_result['risk_category']}")
        print(f"   Latency: {(time.perf_counter() - start)*1000:.2f}ms")

        # 2. Context Retrieval (Optional for this test, but good to check)
        # In real flow, this happens in parallel or before reasoning

        # 3. Reasoning
        print("\n2️⃣  LLM Reasoning...")
        start = time.perf_counter()
        reasoning_req = {
            "event_id": EVENT["event_id"],
            "content": EVENT["content"],
            "risk_score": screening_result["risk_score"],
            "flags": screening_result["flags"],
            "similar_events": [] # contextual data would go here
        }

        resp = await client.post(f"{REASONING_URL}/v1/reason", json=reasoning_req)
        print(f"   Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"❌ Reasoning failed: {resp.text}")
            return

        reasoning_result = resp.json()
        print(f"   Harm Score: {reasoning_result['harm_score']}")
        print(f"   Explanation: {reasoning_result['explanation']}")
        print(f"   Recommended Actions: {reasoning_result['recommended_actions']}")
        print(f"   Latency: {(time.perf_counter() - start)*1000:.2f}ms")

        # 4. Action Execution
        print("\n3️⃣  Action Execution...")
        start = time.perf_counter()
        action_req = {
            "event_id": EVENT["event_id"],
            "content_id": "msg-123", # placeholder
            "author_id": EVENT["author_id"],
            "harm_score": reasoning_result["harm_score"],
            "recommended_actions": reasoning_result["recommended_actions"],
            "explanation": reasoning_result["explanation"]
        }

        resp = await client.post(f"{EXECUTOR_URL}/v1/execute", json=action_req)
        print(f"   Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"❌ Execution failed: {resp.text}")
            return

        execution_result = resp.json()
        print(f"   Executed: {[a['action'] + ': ' + ('✅' if a['success'] else '❌') for a in execution_result['executed_actions']]}")
        print(f"   Latency: {(time.perf_counter() - start)*1000:.2f}ms")

    print("\n✅ End-to-End Test Complete!")

if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(run_pipeline())
