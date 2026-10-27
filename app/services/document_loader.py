import asyncio
from pathlib import Path
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import get_settings


def _extract_pages_sync(file_path: Path) -> list[Document]:
    """Extrae sincrónicamente texto y metadatos de un archivo PDF usando PyMuPDF."""
    if not file_path.exists():
        raise FileNotFoundError(f"Documento no encontrado: {file_path}")

    documents: list[Document] = []
    with fitz.open(file_path) as pdf_doc:
        total_pages = len(pdf_doc)
        for page_num in range(total_pages):
            page = pdf_doc[page_num]
            text = page.get_text("text").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "file_path": str(file_path.resolve()),
                            "page": page_num + 1,
                            "total_pages": total_pages,
                        },
                    )
                )
    return documents


async def load_and_extract_text(file_path: str | Path) -> list[Document]:
    """Carga y extrae texto de un PDF de forma asíncrona sin bloquear el event loop."""
    path = Path(file_path)
    return await asyncio.to_thread(_extract_pages_sync, path)


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Divide documentos en chunks preservando el contexto clínico y metadatos."""
    settings = get_settings()
    size = chunk_size or settings.CHUNK_SIZE
    overlap = chunk_overlap or settings.CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return text_splitter.split_documents(documents)
