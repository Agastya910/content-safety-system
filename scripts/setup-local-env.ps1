# Set environment variables for local services
$env:REDIS_HOST = "localhost"
$env:REDIS_PORT = "6379"
$env:REDIS_PASSWORD = "redis_dev_password"
$env:QDRANT_HOST = "localhost"
$env:QDRANT_PORT = "6333"
$env:QDRANT_API_KEY = "qdrant_dev_key_unsafe_only"
$env:DATABASE_URL = "postgresql+asyncpg://safety:dev_password_unsafe_only@localhost:5432/safety_db"
$env:LOG_LEVEL = "DEBUG"

Write-Host "✅ Environment variables set for local development" -ForegroundColor Green
Write-Host ""
Write-Host "To start services, run in separate terminals:" -ForegroundColor Cyan
Write-Host "1. Risk Screening:" -ForegroundColor Yellow
Write-Host "   cd services\risk-screening"
Write-Host "   uvicorn src.risk_screening.main:app --host 0.0.0.0 --port 8002 --reload"
Write-Host ""
Write-Host "2. Embedding Context:" -ForegroundColor Yellow
Write-Host "   cd services\embedding-context"
Write-Host "   uvicorn src.embedding_context.main:app --host 0.0.0.0 --port 8003 --reload"
