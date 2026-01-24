from fastapi import FastAPI

app = FastAPI(title="Embedding Context Service")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "embedding-context"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}
