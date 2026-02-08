import asyncio
import httpx
import sys

BASE_URL = "http://localhost:8002"
API_KEY = "test-key"

async def verify():
    print("🔍 Verifying Risk Screening Service Upgrade...\n")

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Check Health & Config
        try:
            resp = await client.get(f"{BASE_URL}/health")
            resp.raise_for_status()
            data = resp.json()
            print(f"✅ Service Health: OK (Model: {data.get('embedding_model')})")
        except Exception as e:
            print(f"❌ Service Health: FAILED ({e})")
            return

        # 2. Check Metrics (Queue & Config)
        try:
            resp = await client.get(f"{BASE_URL}/v1/metrics")
            data = resp.json()
            print(f"✅ Metrics: OK (Queue Depth: {data.get('queue_depth')})")
        except Exception as e:
            print(f"❌ Metrics: FAILED ({e})")

        # 3. Test Seeding (Semantic Match)
        # We send a phrase that is EXACTLY in the seed corpus
        seed_phrase = "You are worthless and should just die."
        print(f"\n🧪 Testing Exact Seed Match: '{seed_phrase}'")
        try:
            resp = await client.post(
                f"{BASE_URL}/v1/risk/screen",
                params={"event_id": "test-seed-1", "content": seed_phrase},
                headers={"api-key": API_KEY}
            )
            result = resp.json()
            score = result['risk_score']
            embedding_score = result['details']['embedding_score']
            print(f"   Result: Score={score:.4f}, EmbScore={embedding_score:.4f}")

            if embedding_score > 0.9:
                print("✅ Semantic Search: WORKING (High Similarity detected)")
            else:
                print("❌ Semantic Search: SUSPICIOUS (Low Similarity for exact match)")

        except Exception as e:
            print(f"❌ Test Failed: {e}")

        # 4. Test Heuristic Only
        heuristic_phrase = "CAPS LOCK CAPS LOCK CAPS LOCK"
        print(f"\n🧪 Testing Heuristic Only: '{heuristic_phrase}'")
        try:
            resp = await client.post(
                f"{BASE_URL}/v1/risk/screen",
                params={"event_id": "test-heur-1", "content": heuristic_phrase},
                headers={"api-key": API_KEY}
            )
            result = resp.json()
            score = result['risk_score']
            heuristic_score = result['details']['heuristic_score']
            print(f"   Result: Score={score:.4f}, HeurScore={heuristic_score:.4f}")
        except Exception as e:
            print(f"❌ Test Failed: {e}")

        print("\nVerification Complete.")

if __name__ == "__main__":
    asyncio.run(verify())
