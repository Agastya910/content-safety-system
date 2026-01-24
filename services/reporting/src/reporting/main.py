from fastapi import FastAPI

app = FastAPI(title="Reporting Service")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "reporting"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}
