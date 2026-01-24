from fastapi import FastAPI

app = FastAPI(title="Action Executor Service")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "action-executor"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}
