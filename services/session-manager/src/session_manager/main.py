from fastapi import FastAPI

app = FastAPI(title="Session Manager Service")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "session-manager"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}
