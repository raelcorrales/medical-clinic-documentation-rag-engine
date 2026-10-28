# MASTER PROMPT: Generator for the Asynchronous Medical & Operational RAG Engine

> **Usage Instructions:** Copy and paste the following prompt block into **Claude Code**, **Antigravity**, **Cursor**, or **Windsurf** to recreate this entire project from scratch in an empty repository.

---

```markdown
Act as a Senior Software Engineer specializing in Asynchronous Backend Systems (Python / FastAPI) and Artificial Intelligence Architecture for the Healthcare and Clinical sector.

Your goal is to build an end-to-end, fully functional **Asynchronous RAG Engine for Clinical Documentation, Laboratory Analyzer Error Codes, and Normative Guidelines (ISO 15189 / NOM-007 / NOM-004)**, designed for production-scale deployment and optimized to run with low-cost, local open-weight models (Ollama + Elasticsearch) with seamless cloud extensibility.

---

## 1. 🛠️ Tech Stack & Architecture

* **Language:** Python 3.12+ (compatible with Python 3.14).
* **Web Framework:** FastAPI (100% async, CORS enabled, Pydantic v2 validation).
* **Vector Search Engine:** Elasticsearch 8.15+ (dense HNSW vector search deployed via `docker-compose.yml`), decoupled with a Factory pattern supporting fallback to ChromaDB.
* **LLM & Embedding Models:**
  - Primary / Local (zero token cost): **Ollama** (`llama3.2` for structured clinical inference and `nomic-embed-text` for 768-dimensional embeddings).
  - Production / Self-Hosted: **vLLM** / **OpenAI-compatible** custom endpoints (`/v1`).
  - Cloud Fallback: **OpenAI** (`gpt-4o-mini` and `text-embedding-3-small`), switchable via `.env`.
* **Document Parsing:** `PyMuPDF` (`fitz`), executed asynchronously with `asyncio.to_thread` to prevent event-loop blocking.
* **RAG Orchestration:** LangChain with LCEL (LangChain Expression Language).
* **Testing:** `pytest` with `pytest-asyncio` and integration mocks.

---

## 2. 📁 Project Structure

Create the following modular architecture:

```text
rag-async-doc-medical/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app, CORS, lifespan handler, and /health
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py               # Pydantic BaseSettings (.env)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_loader.py      # Async PDF extraction & chunking preserving page metadata
│   │   ├── llm_factory.py          # Dynamic factory for LLMs & Embeddings (Ollama / vLLM / OpenAI)
│   │   ├── vector_store.py         # ElasticsearchStore (HNSW) / Chroma interface
│   │   └── rag_engine.py           # LCEL RAG chain with strict clinical prompt
│   └── api/
│       ├── __init__.py
│       └── v1/
│           ├── __init__.py
│           └── rag.py              # POST /ingest and POST /query endpoints
├── tests/
│   ├── __init__.py
│   ├── test_document_loader.py     # Unit tests for parsing and chunking
│   ├── test_api.py                 # Integration tests for FastAPI (/health, /ingest, /query)
│   └── data/                       # Synthetic clinical test PDFs
├── .env.example
├── .gitignore
├── docker-compose.yml              # Single-node Elasticsearch 8.15 with volume persistence
├── pytest.ini                      # Pythonpath & asyncio configuration
├── requirements.txt
└── README.md
```

---

## 3. 📋 Step-by-Step Implementation Guide

### Step 1: Infrastructure & Environment Setup
1. Generate `docker-compose.yml` to run **Elasticsearch 8.15** in `single-node` mode, with security disabled for local development, exposing port `9200`, and mounting the persistent volume `es_data`.
2. Generate `requirements.txt` containing: `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `python-multipart`, `pymupdf`, `langchain`, `langchain-core`, `langchain-community`, `langchain-ollama`, `langchain-elasticsearch`, `elasticsearch`, `httpx`, `pytest`, `pytest-asyncio`.
3. Create `app/core/config.py` using `pydantic-settings` configuring:
   - `LLM_PROVIDER` (`ollama` | `openai` | `vllm` | `openai_compatible`), `EMBEDDING_PROVIDER`.
   - `OLLAMA_BASE_URL` (e.g. `http://localhost:11434` or remote IP), `OLLAMA_MODEL` (`llama3.2`), `OLLAMA_EMBED_MODEL` (`nomic-embed-text`).
   - `CUSTOM_LLM_BASE_URL`, `CUSTOM_LLM_MODEL`, `CUSTOM_EMBED_BASE_URL` for vLLM clusters.
   - `VECTOR_STORE_BACKEND` (`elasticsearch` | `chroma`), `ELASTICSEARCH_URL` (`http://localhost:9200`), `ELASTICSEARCH_INDEX` (`clinical_docs`).
   - RAG parameters: `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`, `TOP_K=4`.
4. Provide pre-configured `.env.example` and `.env` files.

### Step 2: Clinical Document ETL Pipeline (`app/services/document_loader.py`)
1. Implement `load_and_extract_text(file_path)`: Read PDFs using `PyMuPDF` asynchronously via `asyncio.to_thread`, extracting text page-by-page and tagging metadata: `source` (filename), `page` (1-indexed page number), and `total_pages`.
2. Implement `chunk_documents(documents, chunk_size, chunk_overlap)` with LangChain's `RecursiveCharacterTextSplitter`. Ensure clinical context and page metadata persist across every generated chunk.

### Step 3: Model Factory and Vector Store Interface
1. In `app/services/llm_factory.py`: Implement `get_llm()` and `get_embeddings()` dynamically resolving between Ollama, OpenAI, and vLLM / OpenAI-compatible endpoints.
2. In `app/services/vector_store.py`:
   - Implement `embed_and_index_chunks(chunks)` using `ElasticsearchStore` (with approximate HNSW vector retrieval). Support optional cluster credentials (`es_api_key` or `es_user`/`es_password`).
   - Implement `search_similar_chunks(query, k)` and `get_retriever()`.

### Step 4: High-Value Clinical RAG Chain (`app/services/rag_engine.py`)
1. Design the system prompt (`CLINICAL_RAG_PROMPT`). Focus on high-value operational troubleshooting rather than trivial facts. **Strict guidelines:**
   - Answer **exclusively** from the retrieved context. If information is missing, output verbatim: *"La información solicitada no se encuentra disponible en la documentación provista."*
   - Mandatory structured output across 3 pillars:
     * **Immediate Required Action:** Sequential, concrete operational steps.
     * **Technical / Normative Criterion / Error Code:** Time constraints, safety parameters, reagents, or applicable standards (ISO 15189, NOM-007, NOM-004).
     * **Mandatory Traceability & Documentation:** Specific logbooks, incident forms, or required signatures.
2. Build the LCEL pipeline: `retriever | format_context | prompt | llm | StrOutputParser()`.
3. Return a Pydantic `RAGResult` schema containing `query`, `answer`, and a list of `sources` (`source`, `page`, and `snippet`).

### Step 5: FastAPI Endpoints (`app/api/v1/rag.py` and `app/main.py`)
1. `POST /api/v1/rag/ingest`:
   - Accept `UploadFile = File(...)` validating `.pdf` format.
   - Save to a temporary file, extract pages, chunk, index embeddings in Elasticsearch, and respond with a summary (`filename`, `total_pages`, `chunks_indexed`).
2. `POST /api/v1/rag/query`:
   - Accept JSON `{"question": "...", "top_k": 4}`.
   - Execute the RAG chain and return `RAGResult`.
3. `app/main.py`: Configure CORS, a `/health` probe, and register `/api/v1/rag`.

### Step 6: Test Data Generation and Test Suite (`tests/`)
1. Create a script or fixture generating 3 realistic clinical PDFs in `tests/data/`:
   - `manual_codigos_error_equipos.pdf`: Analyzer error codes (e.g. Cobas c501 `E-204` clot detection, probe wash, manual purge, and `BIT-EQ-02` logbook).
   - `protocolo_contingencia_seguridad.pdf`: Centrifuge aerosol biohazard spills under ISO 15189 / NOM-007 (30-min evacuation, N95/FFP2 PPE, 0.5% hypochlorite with 30-min contact time).
   - `trazabilidad_legal_escalamiento.pdf`: Minute-by-minute legal escalation algorithm (15-min hard deadline to Medical Subdirection) and `F-VAL-03` logbook under NOM-004.
2. Write unit tests in `tests/test_document_loader.py` and API integration tests in `tests/test_api.py`.
3. Ensure the test suite passes 100% with `pytest -v`.

---

## 4. 🎯 Definition of Done (DoD)

* [ ] Services start cleanly with `docker compose up -d` and `uvicorn app.main:app --reload`.
* [ ] PDF uploads via `/api/v1/rag/ingest` successfully index vectors into Elasticsearch.
* [ ] Queries to `/api/v1/rag/query` return structured, strictly grounded answers with source citations (`source`, `page`, `snippet`).
* [ ] Zero hallucinations when queries fall outside the provided documentation.
* [ ] Full `pytest` test suite runs completely green.
```
