import fitz  # PyMuPDF
import pytest
from pathlib import Path
from langchain_core.documents import Document
from app.services.document_loader import load_and_extract_text, chunk_documents


def create_dummy_pdf(file_path: Path, text: str) -> None:
    """Genera un archivo PDF simple con PyMuPDF para pruebas unitarias."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    doc.save(file_path)
    doc.close()


@pytest.mark.asyncio
async def test_load_and_extract_text(tmp_path: Path):
    pdf_path = tmp_path / "test_pno.pdf"
    content = "Procedimiento Normalizado de Operacion PNO-002: Calibracion de espectrofotometro."
    create_dummy_pdf(pdf_path, content)

    docs = await load_and_extract_text(pdf_path)

    assert len(docs) == 1
    assert "PNO-002" in docs[0].page_content
    assert docs[0].metadata["source"] == "test_pno.pdf"
    assert docs[0].metadata["page"] == 1


def test_chunk_documents_preserves_metadata():
    long_text = "Seccion 1: Control de calidad en laboratorio clinico. " * 50
    doc = Document(
        page_content=long_text,
        metadata={"source": "guia_calidad.pdf", "page": 3},
    )

    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "guia_calidad.pdf"
        assert chunk.metadata["page"] == 3
        assert len(chunk.page_content) <= 250  # Margen por separadores
