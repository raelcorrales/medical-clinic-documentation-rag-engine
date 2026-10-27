import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field

from app.services.document_loader import load_and_extract_text, chunk_documents
from app.services.vector_store import embed_and_index_chunks
from app.services.rag_engine import generate_rag_response, RAGResult

router = APIRouter(prefix="/rag", tags=["RAG Medical Engine"])


class IngestResponse(BaseModel):
    message: str
    filename: str
    total_pages: int
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Pregunta clínica o consulta sobre PNO/ISO")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Cantidad de fragmentos a recuperar")


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Subir e indexar un documento clínico en PDF",
)
async def ingest_document(file: UploadFile = File(...)):
    """Recibe un archivo PDF clínico (PNO, norma, guía), extrae su texto, divide en chunks

    e indexa sus embeddings en Elasticsearch.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se admiten archivos en formato PDF.",
        )

    # Guardar en archivo temporal para PyMuPDF
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = Path(tmp_file.name)
        shutil.copyfileobj(file.file, tmp_file)

    try:
        # 1. Extraer páginas de forma asíncrona
        documents = await load_and_extract_text(tmp_path)
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="El PDF no contiene texto extraíble.",
            )

        # Ajustar nombre de la fuente en los metadatos al nombre real del archivo
        for doc in documents:
            doc.metadata["source"] = file.filename
            doc.metadata["file_path"] = file.filename

        # 2. Dividir en chunks con preservación clínica
        chunks = chunk_documents(documents)

        # 3. Vectorizar e indexar en Elasticsearch
        chunks_count = embed_and_index_chunks(chunks)

        total_pages = documents[-1].metadata.get("total_pages", len(documents))

        return IngestResponse(
            message="Documento procesado e indexado exitosamente.",
            filename=file.filename,
            total_pages=total_pages,
            chunks_indexed=chunks_count,
        )
    finally:
        # Limpieza de archivo temporal
        if tmp_path.exists():
            tmp_path.unlink()


@router.post(
    "/query",
    response_model=RAGResult,
    status_code=status.HTTP_200_OK,
    summary="Consultar el motor RAG sobre la documentación clínica",
)
async def query_rag(request: QueryRequest):
    """Consulta la base documental médica mediante recuperación vectorial y generación con LLM."""
    try:
        result = await generate_rag_response(
            question=request.question,
            top_k=request.top_k,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la consulta RAG: {str(e)}",
        )
