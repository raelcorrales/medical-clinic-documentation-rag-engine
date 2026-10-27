# TASK: 3. Motor RAG Asíncrono sobre Documentación Médica/Clínica

## 📌 Contexto
Este proyecto tiene como objetivo construir un sistema de Recuperación Aumentada por Generación (RAG) especializado en procesar e interrogar documentación clínica normativa (como Procedimientos Normalizados de Operación o lineamientos ISO 15189)[cite: 7]. Se enfoca en integrar las capacidades de ingeniería de datos y desarrollo backend asíncrono con modelos fundacionales (LLMs) y bases de datos vectoriales.

## 🛠️ Tech Stack
*   **Lenguaje:** Python 3.12+
*   **Framework Web:** FastAPI (async)
*   **Framework LLM/Agentes:** LangChain
*   **Base de Datos Vectorial:** ChromaDB o OpenSearch local (vía Docker)
*   **Modelos AI:** OpenAI API (GPT-4o / text-embedding-3) o Google Vertex AI[cite: 8, 18]
*   **Procesamiento de PDF:** `PyMuPDF` o `pdfplumber`
*   **Testing:** `pytest`

---

## 📋 Checklist de Implementación

### Fase 1: Configuración del Entorno y Base Vectorial
- [x] Inicializar una nueva rama o directorio para el Motor RAG.
- [x] Instalar dependencias clave: `fastapi`, `langchain`, `langchain-ollama`, `langchain-elasticsearch`, `elasticsearch`, `pymupdf`.
- [x] Configurar el `docker-compose.yml` para incluir una instancia de Elasticsearch 8.x lista para millones de documentos.
- [x] Gestionar las claves y endpoints (Ollama remoto / Elasticsearch) utilizando variables de entorno y `pydantic-settings`.

### Fase 2: Pipeline de Ingesta y Procesamiento de Documentos (ETL)
- [x] Crear el módulo `app/services/document_loader.py`.
- [x] Implementar la función asíncrona `load_and_extract_text(file_path: str)` para leer archivos PDF (PNO o guías clínicas con PyMuPDF).
- [x] Implementar la función `chunk_documents(documents: list)` utilizando `RecursiveCharacterTextSplitter` de LangChain preservando metadatos (página, fuente).
- [x] Crear el servicio `app/services/vector_store.py`.
- [x] Implementar la función `embed_and_index_chunks(chunks: list)` que utilice el modelo de *embeddings* seleccionado (`nomic-embed-text` vía Ollama) para vectorizar los fragmentos y almacenarlos en Elasticsearch.

### Fase 3: Cadena de Recuperación y Generación (RAG Chain)
- [x] Crear el módulo `app/services/rag_engine.py`.
- [x] Configurar el recuperador (*Retriever*) para consultar Elasticsearch buscando los *chunks* más relevantes (*Top-K*).
- [x] Diseñar el *Prompt Template* para restringir al modelo. **Instrucción crítica:** El modelo responde exclusivamente basándose en el contexto recuperado y reporta si la información no se encuentra en el documento.
- [x] Ensamblar la cadena RAG utilizando LCEL (LangChain Expression Language) integrando el recuperador, el *prompt* y el LLM (`llama3.2` vía Ollama).

### Fase 4: Endpoints de la API (FastAPI)
- [x] Configurar el router en `app/api/v1/rag.py` y montar en `app/main.py`.
- [x] Implementar `POST /api/v1/rag/ingest`: Endpoint para subir un archivo PDF, procesarlo y cargar los *embeddings* en la base vectorial.
- [x] Implementar `POST /api/v1/rag/query`: Endpoint que reciba una pregunta en JSON, ejecute la cadena RAG y retorne la respuesta junto con los fragmentos de texto fuente (*source documents*) utilizados.

### Fase 5: Pruebas y Validación
- [x] Escribir pruebas unitarias para validar que el *text splitter* divide correctamente los documentos sin perder información crítica ni metadatos.
- [x] Escribir una prueba de integración para el endpoint de consulta (`/api/v1/rag/query`), mockeando la llamada al LLM externo para asegurar que la cadena se ejecuta correctamente.
- [x] Ingestar un documento clínico de prueba (PNO-LAB-014 de Valores Críticos bajo ISO 15189) y verificar cualitativamente la exactitud de las respuestas.

---

## 🎯 Definición de Terminado (Definition of Done - DoD)
- [x] El sistema permite subir un documento PDF a través de la API y este es indexado exitosamente en la base de datos vectorial.
- [x] Las consultas al endpoint de *query* retornan respuestas coherentes, limitadas al contexto del documento proveído.
- [x] La respuesta de la API incluye los metadatos de los documentos fuente (ej. nombre del archivo, número de página) para permitir la trazabilidad de la información.
- [x] El código está modularizado, tipado y validado mediante pruebas locales.