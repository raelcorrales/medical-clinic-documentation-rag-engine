# PROMPT MAESTRO: Generador del Motor RAG Asíncrono para Documentación Médica y Operativa

> **Instrucciones de uso:** Copia y pega el siguiente bloque de texto en **Claude Code**, **Antigravity**, **Cursor** o **Windsurf** para recrear este proyecto desde cero en un repositorio vacío.

---

```markdown
Actúa como un Ingeniero de Software Senior especializado en Backend Asíncrono (Python / FastAPI) y Arquitectura de Inteligencia Artificial para el sector Salud / Clínico.

Tu objetivo es construir de forma integral y autónoma un **Motor RAG Asíncrono sobre Documentación Médica, Códigos de Error de Analizadores y Normativa Clínica (ISO 15189 / NOM-007 / NOM-004)**, diseñado para escalar a nivel productivo y optimizado para ejecutarse con modelos locales open-weights y de bajo costo (Ollama + Elasticsearch).

---

## 1. 🛠️ Tech Stack & Arquitectura

* **Lenguaje:** Python 3.12+ (compatible con Python 3.14).
* **Framework Web:** FastAPI (100% asíncrono, CORS configurado, validación con Pydantic v2).
* **Motor de Búsqueda Vectorial:** Elasticsearch 8.15+ (con búsqueda vectorial densa HNSW desplegado vía `docker-compose.yml`), con diseño desacoplado (patrón Factory) que admita fallback a ChromaDB.
* **Modelos LLM & Embeddings:** 
  - Primario / Local (costo cero de tokens): **Ollama** (`llama3.2` para inferencia clínica estructurada y `nomic-embed-text` para embeddings vectoriales de 768 dimensiones).
  - Secundario / Cloud: **OpenAI** (`gpt-4o-mini` y `text-embedding-3-small`), conmutable vía `.env`.
* **Procesamiento de Documentos:** `PyMuPDF` (`fitz`), ejecutado asíncronamente mediante `asyncio.to_thread` para no bloquear el event loop.
* **Orquestación RAG:** LangChain con LCEL (LangChain Expression Language).
* **Testing:** `pytest` con `pytest-asyncio` y mocks de integración.

---

## 2. 📁 Estructura del Proyecto

Crea la siguiente estructura modular:

```text
rag-async-doc-medical/
├── app/
│   ├── __init__.py
│   ├── main.py                     # Instancia FastAPI, middlewares, ciclo de vida lifespan y /health
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py               # Settings con pydantic-settings (.env)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_loader.py      # Extracción asíncrona de PDFs y chunking con metadatos por página
│   │   ├── llm_factory.py          # Factoría para instanciar LLMs y Embeddings (Ollama / OpenAI)
│   │   ├── vector_store.py         # Interfaz con ElasticsearchStore (HNSW) / Chroma
│   │   └── rag_engine.py           # Cadena RAG (LCEL) con Prompt Clínico Estricto
│   └── api/
│       ├── __init__.py
│       └── v1/
│           ├── __init__.py
│           └── rag.py              # Endpoints POST /ingest y POST /query
├── tests/
│   ├── __init__.py
│   ├── test_document_loader.py     # Pruebas unitarias de parsing y chunking
│   ├── test_api.py                 # Pruebas de integración para FastAPI (/health, /ingest, /query)
│   └── data/                       # PDFs clínicos sintéticos de prueba
├── .env.example
├── .gitignore
├── docker-compose.yml              # Servicio single-node de Elasticsearch 8.15 con persistencia
├── pytest.ini                      # Configuración de pythonpath y asyncio_mode
├── requirements.txt
└── README.md
```

---

## 3. 📋 Requisitos de Implementación Paso a Paso

### Paso 1: Configuración de Infraestructura y Entorno
1. Genera un `docker-compose.yml` para desplegar **Elasticsearch 8.15** en modo `single-node`, sin seguridad SSL estricta para desarrollo local, exponiendo el puerto `9200` y con volumen persistente `es_data`.
2. Genera `requirements.txt` con: `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `python-multipart`, `pymupdf`, `langchain`, `langchain-core`, `langchain-community`, `langchain-ollama`, `langchain-elasticsearch`, `elasticsearch`, `httpx`, `pytest`, `pytest-asyncio`.
3. Crea `app/core/config.py` usando `pydantic-settings` para parametrizar:
   - `LLM_PROVIDER` (`ollama` | `openai`), `EMBEDDING_PROVIDER` (`ollama` | `openai`).
   - `OLLAMA_BASE_URL` (ej. `http://localhost:11434` o IP remota), `OLLAMA_MODEL` (`llama3.2`), `OLLAMA_EMBED_MODEL` (`nomic-embed-text`).
   - `VECTOR_STORE_BACKEND` (`elasticsearch` | `chroma`), `ELASTICSEARCH_URL` (`http://localhost:9200`), `ELASTICSEARCH_INDEX` (`clinical_docs`).
   - Parámetros RAG: `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`, `TOP_K=4`.
4. Crea `.env.example` y `.env` pre-poblados.

### Paso 2: Pipeline ETL de Documentos Clínicos (`app/services/document_loader.py`)
1. Implementa `load_and_extract_text(file_path)`: Debe leer PDFs usando `PyMuPDF` de forma no bloqueante (`asyncio.to_thread`), extrayendo el texto página por página e inyectando metadatos: `source` (nombre del archivo), `page` (número de página 1-indexed) y `total_pages`.
2. Implementa `chunk_documents(documents, chunk_size, chunk_overlap)` usando `RecursiveCharacterTextSplitter` de LangChain. Garantiza que los metadatos de trazabilidad persistan en cada chunk.

### Paso 3: Factoría de Modelos y Vector Store
1. En `app/services/llm_factory.py`: Implementa `get_llm()` y `get_embeddings()` dinámicamente según la configuración de `pydantic-settings`.
2. En `app/services/vector_store.py`:
   - Implementa `embed_and_index_chunks(chunks)` utilizando `ElasticsearchStore` (estrategia de búsqueda vectorial aproximada HNSW).
   - Implementa `search_similar_chunks(query, k)` y `get_retriever()`.

### Paso 4: Cadena RAG de Alto Valor Clínico (`app/services/rag_engine.py`)
1. Diseña el prompt del sistema (`CLINICAL_RAG_PROMPT`). Debe evitar preguntas triviales y enfocarse en resolución operativa. **Reglas críticas:**
   - Responder **exclusivamente** con el contexto provisto. Si falta información, declarar textualmente: *"La información solicitada no se encuentra disponible en la documentación provista."*
   - Salida estructurada obligatoria en 3 ejes:
     * **Acción Inmediata Requerida:** Instrucciones operativas secuenciales.
     * **Criterio Técnico / Normativo / Código de Error:** Tiempos límite, parámetros de seguridad, reactivos o normas aplicables (ej. ISO 15189, NOM-007, NOM-004).
     * **Registro y Trazabilidad Obligatoria:** Formato, bitácora o firmas requeridas.
2. Ensambla la cadena LCEL: `retriever | format_context | prompt | llm | StrOutputParser()`.
3. Retorna un esquema Pydantic `RAGResult` que contenga `query`, `answer` y una lista de `sources` con `source`, `page` y `snippet`.

### Paso 5: Endpoints FastAPI (`app/api/v1/rag.py` y `app/main.py`)
1. `POST /api/v1/rag/ingest`:
   - Acepta `UploadFile = File(...)` (validando que sea PDF).
   - Guarda en archivo temporal, extrae páginas, divide en chunks, indexa en Elasticsearch y responde con resumen (`filename`, `total_pages`, `chunks_indexed`).
2. `POST /api/v1/rag/query`:
   - Acepta JSON `{"question": "...", "top_k": 4}`.
   - Ejecuta la cadena RAG y responde con `RAGResult`.
3. `app/main.py`: Monta CORS, endpoint de salud `/health`, y el router `/api/v1/rag`.

### Paso 6: Generación de Documentación de Prueba y Tests (`tests/`)
1. Crea un script o fixture que genere 3 documentos PDF clínicos realistas en `tests/data/`:
   - `manual_codigos_error_equipos.pdf`: Códigos de error de analizadores (ej. Cobas c501 `E-204` clot detection, purga y bitácora `BIT-EQ-02`).
   - `protocolo_contingencia_seguridad.pdf`: Derrames biológicos en centrífuga con riesgo de aerosoles bajo ISO 15189 / NOM-007 (evacuación de 30 min, EPP N95/FFP2, hipoclorito al 0.5% con 30 min de contacto).
   - `trazabilidad_legal_escalamiento.pdf`: Algoritmo de escalamiento legal minuto a minuto (15 min bloqueo legal a Subdirección Médica) y bitácora `F-VAL-03` bajo NOM-004.
2. Escribe pruebas unitarias en `tests/test_document_loader.py` y pruebas de integración con mocks en `tests/test_api.py`.
3. Asegura que la suite de `pytest -v` pase al 100%.

---

## 4. 🎯 Criterios de Aceptación (Definition of Done)

* [ ] El servicio levanta con `docker compose up -d` y `uvicorn app.main:app --reload`.
* [ ] La subida de PDFs vía `/api/v1/rag/ingest` indexa los vectores en Elasticsearch.
* [ ] Las consultas a `/api/v1/rag/query` devuelven respuestas estrictamente apegadas al documento, estructuradas y con metadatos de fuentes citadas (`source`, `page`, `snippet`).
* [ ] Cero alucinaciones si se pregunta algo fuera del contexto.
* [ ] La suite de pruebas de `pytest` pasa en verde sin errores.
```
