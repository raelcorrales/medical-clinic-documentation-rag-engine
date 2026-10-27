# Motor RAG Asíncrono sobre Documentación Médica / Clínica

Sistema de Recuperación Aumentada por Generación (RAG) asíncrono para interrogar documentación clínica normativa (Procedimientos Normalizados de Operación - PNOs, guías de laboratorio e ISO 15189), garantizando estricta adherencia al contexto y trazabilidad de fuentes.

---

## 🛠️ Stack Tecnológico

- **Framework Web:** [FastAPI](https://fastapi.tiangolo.com/) (Asíncrono)
- **Base Vectorial:** [Elasticsearch 8.15](https://www.elastic.co/) (Búsqueda vectorial dense HNSW en Docker)
- **Modelos LLM & Embeddings:** [Ollama](https://ollama.com/) en red local (`nomic-embed-text` + `llama3.2`) con fallback a OpenAI
- **Procesamiento de PDF:** [PyMuPDF](https://pymupdf.readthedocs.io/)
- **Orquestación RAG:** [LangChain](https://www.langchain.com/) (LCEL)
- **Testing:** [pytest](https://docs.pytest.org/) con `pytest-asyncio`

---

## 🚀 Inicio Rápido

### 1. Requisitos Previos
- Docker Desktop activo
- Servidor Ollama disponible en `http://192.168.3.4:11434` (o local) con los modelos `llama3.2` y `nomic-embed-text`

### 2. Iniciar Elasticsearch
```bash
docker compose up -d
```

### 3. Activar Entorno Virtual e Instalar Dependencias
```bash
# En Windows:
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Ejecutar la API
```bash
.\.venv\Scripts\uvicorn app.main:app --reload --port 8000
```
- Documentación Swagger interactiva: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## 📡 Endpoints de la API

### Ingesta de Documentos PDF
- **Ruta:** `POST /api/v1/rag/ingest`
- **Body:** `multipart/form-data` con campo `file` (.pdf)
- **Respuesta:**
```json
{
  "message": "Documento procesado e indexado exitosamente.",
  "filename": "pno_valores_criticos.pdf",
  "total_pages": 1,
  "chunks_indexed": 1
}
```

### Consulta RAG Clínica
- **Ruta:** `POST /api/v1/rag/query`
- **Body:**
```json
{
  "question": "¿Cuáles son los valores de pánico para el potasio sérico?",
  "top_k": 4
}
```
- **Respuesta:**
```json
{
  "query": "¿Cuáles son los valores de pánico para el potasio sérico?",
  "answer": "Según el contexto clínico proporcionado, los valores de pánico para el potasio sérico son: Menor de 2.8 mEq/L o Mayor de 6.2 mEq/L (riesgo inmediato de arritmia letal).",
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

## 🧪 Pruebas Automatizadas

```bash
.\.venv\Scripts\pytest -v
```
