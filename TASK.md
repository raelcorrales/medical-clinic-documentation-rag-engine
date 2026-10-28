# TASK: 3. Asynchronous RAG Engine over Medical/Clinical Documentation

## 📌 Context
The goal of this project is to build an asynchronous Retrieval-Augmented Generation (RAG) system specialized in processing and querying normative clinical documentation (such as Standard Operating Procedures - SOPs or ISO 15189 guidelines). It focuses on integrating data engineering and asynchronous backend development with foundation models (LLMs) and vector databases.

## 🛠️ Tech Stack
*   **Language:** Python 3.12+ (tested on Python 3.14)
*   **Web Framework:** FastAPI (async)
*   **LLM/Agent Framework:** LangChain (LCEL)
*   **Vector Database:** Elasticsearch 8.15+ (via Docker with dense HNSW vector search) / ChromaDB
*   **AI Models:** Ollama (`llama3.2` / `nomic-embed-text`) / OpenAI API (`gpt-4o-mini` / `text-embedding-3-small`) / vLLM compatible
*   **PDF Processing:** `PyMuPDF` (`fitz`)
*   **Testing:** `pytest` & `pytest-asyncio`

---

## 📋 Implementation Checklist

### Phase 1: Environment Setup and Vector Base
- [x] Initialize a branch/directory for the RAG Engine.
- [x] Install core dependencies: `fastapi`, `langchain`, `langchain-ollama`, `langchain-elasticsearch`, `elasticsearch`, `pymupdf`.
- [x] Configure `docker-compose.yml` to launch an Elasticsearch 8.x instance ready for scale.
- [x] Manage API keys and endpoints (remote Ollama / Elasticsearch) using environment variables and `pydantic-settings`.

### Phase 2: Document Ingestion and Processing Pipeline (ETL)
- [x] Create the `app/services/document_loader.py` module.
- [x] Implement the asynchronous function `load_and_extract_text(file_path: str)` to read PDF files (SOP manuals or clinical guidelines with PyMuPDF).
- [x] Implement the `chunk_documents(documents: list)` function using LangChain's `RecursiveCharacterTextSplitter`, preserving page and source metadata.
- [x] Create the `app/services/vector_store.py` service.
- [x] Implement the `embed_and_index_chunks(chunks: list)` function using the configured embeddings model (`nomic-embed-text` via Ollama) to vectorize and index chunks in Elasticsearch.

### Phase 3: Retrieval and Generation Chain (RAG Chain)
- [x] Create the `app/services/rag_engine.py` module.
- [x] Configure the Retriever to query Elasticsearch for the most relevant chunks (*Top-K*).
- [x] Design the *Prompt Template* to enforce zero-hallucination. **Critical instruction:** The model responds exclusively based on retrieved context and explicitly reports when information is not found in the documents.
- [x] Assemble the RAG chain using LCEL (LangChain Expression Language), integrating the retriever, prompt template, and LLM (`llama3.2` via Ollama).

### Phase 4: API Endpoints (FastAPI)
- [x] Configure router in `app/api/v1/rag.py` and mount in `app/main.py`.
- [x] Implement `POST /api/v1/rag/ingest`: Endpoint to upload a PDF file, process it, and index embeddings into the vector database.
- [x] Implement `POST /api/v1/rag/query`: Endpoint receiving a JSON question, executing the RAG chain, and returning the answer alongside source document metadata (*source documents*).

### Phase 5: Testing and Validation
- [x] Write unit tests verifying that the text splitter partitions documents correctly without losing critical clinical information or metadata.
- [x] Write an integration test for the query endpoint (`/api/v1/rag/query`), mocking external LLM calls to ensure the chain runs properly.
- [x] Ingest sample clinical documents (SOP for Critical Panic Values under ISO 15189) and qualitatively verify response accuracy.

---

## 🎯 Definition of Done (DoD)
- [x] The system allows uploading a PDF document via the API and successfully indexing it into the vector database.
- [x] Queries to the query endpoint return coherent answers strictly limited to the provided document context.
- [x] The API response includes source document metadata (file name, page number, snippet) for complete clinical traceability.
- [x] The codebase is modularized, typed, and validated through local tests.