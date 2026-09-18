# AEGIS-RAG: Production-Grade Hybrid RAG & Semantic Caching Pipeline

A high-performance, modular Retrieval-Augmented Generation (RAG) backend engineered with two-stage hybrid search, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, vector semantic caching, and pre/post-inference guardrails.

## 📌 Architecture & Retrieval Funnel

```
User Request
     │
     ▼
[Guardrails: PII / Injection Check]
     │
     ▼
[Semantic Cache (Cosine Similarity)]
     ├── Cache Hit (Similarity ≥ 0.80) ──► Return Sub-20ms Response
     └── Cache Miss
              │
              ▼
     [Two-Stage Hybrid Search Funnel]
              ├── Dense Retrieval (ChromaDB + Vector Embeddings)
              └── Sparse Retrieval (Rank-BM25 Lexical Keyword Search)
              │
              ▼
     [Reciprocal Rank Fusion (RRF)]
              │
              ▼
     [Cross-Encoder Reranker (ms-marco-MiniLM)]
              │
              ▼
     [Guardrails: Groundedness Validation] ──► Final Verified Output
```

---

## 🛠️ Key Features

- **Hybrid Search Engine (`retriever.py`)** — Combines semantic dense embeddings (`all-MiniLM-L6-v2` via ChromaDB) with sparse keyword matching (`Rank-BM25`) for maximum context recall.
- **Precision Reranking (`reranker.py`)** — Uses **Reciprocal Rank Fusion (RRF)** to merge candidate lists across different score distributions, followed by a heavy **Cross-Encoder model** (`ms-marco-MiniLM-L-6-v2`) for exact context alignment.
- **Low-Latency Semantic Cache (`cache.py`)** — Intercepts queries using embedding vector cosine similarity to serve repeated or semantically similar queries under 20ms, bypassing downstream inference.
- **Guardrails & Security (`guardrails.py`)**
  - *Pre-Inference*: Regex-based PII masking (emails, phone numbers) and prompt injection detection.
  - *Post-Inference*: Lexical grounding ratio checks to prevent hallucinations.
- **Production API (`main.py`)** — Fully asynchronous **FastAPI** web service with Pydantic payload validation and interactive OpenAPI/Swagger documentation.

---

## 📁 Repository Structure

```
.
├── retriever.py       # Dense (ChromaDB) + Sparse (BM25) search implementations
├── reranker.py        # RRF fusion logic & Cross-Encoder rescoring
├── cache.py           # Vector cosine similarity cache layer
├── guardrails.py       # Safety, PII redaction, and hallucination checks
├── main.py            # FastAPI application entry point & API routes
└── requirements.txt   # Project dependencies
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-username/production-rag-pipeline.git
cd production-rag-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn chromadb sentence-transformers rank-bm25 numpy pydantic
```

### 2. Running the API Server

```bash
uvicorn main:app --reload --port 8000
```

Access the interactive API documentation at `http://127.0.0.1:8000/docs`.

---

## 🧪 Example API Usage

### Ingest Documents (`POST /ingest`)

```json
{
  "documents": [
    "Employees receive 20 days of paid vacation per year, scheduled via HR portal.",
    "Remote work allows up to 2 days per week working from home with line manager approval."
  ]
}
```

### Query System (`POST /query`)

```json
{
  "query": "How many days can I work from home weekly?"
}
```

**Response:**

```json
{
  "query": "How many days can I work from home weekly?",
  "answer": "Based on company policy: Remote work allows up to 2 days per week working from home with line manager approval.",
  "source": "rag_pipeline",
  "retrieved_contexts": [
    "Remote work allows up to 2 days per week working from home with line manager approval."
  ]
}
```

---

## 🗺️ Roadmap

- [ ] Connect a local LLM (e.g. via Ollama or vLLM) for end-to-end generation
- [ ] Add RAGAS-based evaluation for retrieval and answer quality
- [ ] Add streaming responses
- [ ] Add authentication / rate limiting for production deployment

---

## 📄 License

MIT
