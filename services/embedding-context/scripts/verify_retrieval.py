import asyncio
import httpx
import json
from uuid import uuid4

BASE_URL = "http://localhost:8003"

async def verify():
    print("🔍 Verifying Retrieval + Context Engine...\n")

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Health Check
        try:
            resp = await client.get(f"{BASE_URL}/health")
            print(f"✅ Service Health: {resp.json()}")
        except Exception as e:
            print(f"❌ Service Health: FAILED ({e})")
            return

        # 2. Context Assembly
        payload = {
            "event": {
                "event_id": str(uuid4()),
                "content": "You are worthless and should just die.",
                "user_id": "victim-123",
                "author_id": "actor-999"
            },
            "risk_prediction": None
        }

        print(f"\n🧪 Testing Context Assembly...")
        try:
            resp = await client.post(f"{BASE_URL}/v1/context/assemble", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                print("✅ Context Retrieval: SUCCESS")
                print(f"   - Similar Events Found: {len(data.get('similar_events', []))}")
                print(f"   - Actor Context: {len(data.get('actor_context', []))}")
                print(f"   - Campaign Context: {len(data.get('campaign_context', []))}")

                # Check if we found the seed phrase logic (if abuse_patterns is shared qdrant)
                # Note: 'abuse_patterns' might be empty unless shared with risk-screening or seeded here.
                # In docker-compose, both services point to same 'qdrant' container.
                # 'risk-screening' seeds 'abuse_patterns'.
                # So if risk-screening ran, this should find hits!
                if len(data.get('similar_events', [])) > 0:
                     print("   ✅ Integrated Memory: Found existing patterns from Risk Screening!")
                else:
                     print("   ⚠️ Integrated Memory: No similar events found (Did risk-screening seed yet?)")

            else:
                print(f"❌ Context Retrieval: FAILED ({resp.status_code}) - {resp.text}")
        except Exception as e:
            print(f"❌ Context Retrieval: ERROR ({e})")

    print("\nVerification Complete.")

if __name__ == "__main__":
    asyncio.run(verify())
