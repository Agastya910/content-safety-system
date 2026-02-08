# Content Safety System (Privacy-First AI)

A local, privacy-preserving content safety platform that uses **Ollama** (LLMs + Embeddings) to detect and act on harmful content in real-time.

![Architecture](https://via.placeholder.com/800x400?text=Safety+System+Architecture)

## 🚀 Features
- **Privacy-First**: All ML runs locally (Ollama). No data leaves your machine.
- **Microservices**: Decoupled FastAPI services for scalability.
- **Real-Time**: Redis Streams for high-throughput event processing.
- **Explainable AI**: LLM reasoning explains *why* content was flagged.
- **Vector Search**: Qdrant-powered semantic search for pattern matching.

## 🛠️ Tech Stack
- **ML/AI**: Ollama (`nomic-embed-text`, `qwen2:7b-instruct`)
- **Backend**: Python 3.12, FastAPI, Uvicorn
- **Data**: Redis (Streams/Cache), PostgreSQL (Relational), Qdrant (Vector)
- **Infrastructure**: Docker (for DBs), Local Process (for Services)

## 📦 Services
1. **Event Ingestion** (`:8001`): Validates and buffers events.
2. **Risk Screening** (`:8002`): Fast heuristic + vector risk scoring.
3. **Embedding Context** (`:8003`): Retreives historical context.
4. **Reasoning** (`:8004`): Deep LLM analysis of intent.
5. **Action Executor** (`:8005`): Applies enforcement (hide, ban, warn).

## ⚡ Quick Start

### Prerequisites
- **Docker Only** installed & running
- **Ollama** installed & running (`ollama serve`)
- **Python 3.12+**

### 1. Setup Environment
```powershell
# Clone repo
git clone https://github.com/Agastya910/content-safety-system.git
cd content-safety-system

# Create virtual env
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r services/risk-screening/requirements.txt
pip install -r services/event-ingestion/requirements.txt
# (or just install shared requirements if consolidated)
```

### 2. Pull ML Models
```powershell
ollama pull nomic-embed-text
ollama pull qwen2:7b-instruct
```

### 3. Start Infrastructure (DBs)
```powershell
docker-compose up -d postgres redis qdrant
```

### 4. Run Services
You can run all services in separate terminals, or use the helper script (coming soon).

**Terminal 1 (Ingestion):**
```powershell
uvicorn services.event_ingestion.src.event_ingestion.main:app --port 8001 --reload
```
*(Repeat for other services on ports 8002-8005)*

### 5. Verify
Run the end-to-end test script:
```powershell
python scripts/test_e2e.py
```

## 📚 Documentation
- [Audit & Status Report](audit.md)
- [Walkthrough](walkthrough.md)
- [Implementation Plan](implementation_plan.md)

## 📄 License
MIT
