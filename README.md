# Asynchronous Clinical / Medical Document RAG Engine

An asynchronous Retrieval-Augmented Generation (RAG) system for querying normative clinical documentation (Standard Operating Procedures - SOPs, laboratory guidelines, and ISO 15189), ensuring strict adherence to context and complete source traceability.

---

## 🛠️ Tech Stack

- **Web Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Fully Asynchronous)
- **Vector Database:** [Elasticsearch 8.15](https://www.elastic.co/) (Dense HNSW vector search running in Docker)
- **LLM & Embedding Models:** [Ollama](https://ollama.com/) on local LAN (`nomic-embed-text` + `llama3.2`) with seamless fallback/extensibility to vLLM and OpenAI
- **PDF Processing:** [PyMuPDF](https://pymupdf.readthedocs.io/)
- **RAG Orchestration:** [LangChain](https://www.langchain.com/) (LCEL)
- **Testing:** [pytest](https://docs.pytest.org/) with `pytest-asyncio`

---

## 🚀 Quickstart

### 1. Prerequisites
- Docker Desktop running
- Ollama server available at `http://192.168.3.4:11434` (or localhost) with `llama3.2` and `nomic-embed-text` models pulled

### 2. Start Elasticsearch
```bash
docker compose up -d
```

### 3. Activate Virtual Environment and Install Dependencies
```bash
# On Windows:
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the API Server
```bash
.\.venv\Scripts\uvicorn app.main:app --reload --port 8000
```
- Interactive Swagger UI documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## 📡 API Endpoints

### PDF Document Ingestion
- **Route:** `POST /api/v1/rag/ingest`
- **Body:** `multipart/form-data` with `file` field (.pdf)
- **Response:**
```json
{
  "message": "Documento procesado e indexado exitosamente.",
  "filename": "pno_valores_criticos.pdf",
  "total_pages": 1,
  "chunks_indexed": 1
}
```

### Clinical RAG Query
- **Route:** `POST /api/v1/rag/query`
- **Body:**
```json
{
  "question": "What are the panic/critical values for serum potassium?",
  "top_k": 4
}
```
- **Response:**
```json
{
  "query": "What are the panic/critical values for serum potassium?",
  "answer": "According to the provided clinical context, the panic values for serum potassium are: Below 2.8 mEq/L or Above 6.2 mEq/L (immediate risk of fatal arrhythmia).",
  "sources": [
    {
      "source": "pno_valores_criticos.pdf",
      "page": 1,
      "snippet": "HOSPITAL CENTRAL - LABORATORIO CLINICO\nPNO-LAB-014: Notificación de Valores Críticos..."
    }
  ]
}
```

---

## 🧪 Automated Testing

```bash
.\.venv\Scripts\pytest -v
```
