# Start all services for Safety System
Write-Host "🚀 Starting Safety System Infrastructure..." -ForegroundColor Cyan

# 1. Start Docker Infrastructure
Write-Host "Step 1: Starting Databases (Docker)..."
docker-compose up -d postgres redis qdrant

# 2. Check Ollama
Write-Host "Step 2: Checking Ollama..."
if (Get-Process ollama -ErrorAction SilentlyContinue) {
    Write-Host "✅ Ollama is running" -ForegroundColor Green
} else {
    Write-Host "⚠️ Ollama not running. Please start 'ollama serve' in a separate terminal." -ForegroundColor Yellow
    exit 1
}

# 3. Create Virtual Env activation command
$venv = ".\.venv\Scripts\activate"

# 4. Start Services in new windows
Write-Host "Step 3: Launching Microservices..."

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd services/event-ingestion; $venv; uvicorn src.event_ingestion.main:app --port 8001 --reload"
Write-Host "Started Event Ingestion (:8001)"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd services/risk-screening; $venv; uvicorn src.risk_screening.main:app --port 8002 --reload"
Write-Host "Started Risk Screening (:8002)"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd services/embedding-context; $venv; uvicorn src.embedding_context.main:app --port 8003 --reload"
Write-Host "Started Embedding Context (:8003)"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd services/reasoning; $venv; uvicorn src.reasoning.main:app --port 8004 --reload"
Write-Host "Started Reasoning Service (:8004)"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd services/action-executor; $venv; uvicorn src.action_executor.main:app --port 8005 --reload"
Write-Host "Started Action Executor (:8005)"

Write-Host "✅ All services launched!" -ForegroundColor Green
Write-Host "Run 'python scripts/test_e2e.py' to verify."
