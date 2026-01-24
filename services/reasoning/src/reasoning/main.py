from fastapi import FastAPI

app = FastAPI(title="Reasoning Service")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "reasoning"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}
