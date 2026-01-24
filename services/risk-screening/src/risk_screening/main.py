from fastapi import FastAPI

app = FastAPI(title="Risk Screening Service")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "risk-screening"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}
