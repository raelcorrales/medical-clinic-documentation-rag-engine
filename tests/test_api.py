import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.rag_engine import RAGResult, SourceDocumentInfo


@pytest.mark.asyncio
async def test_health_check_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "Medical RAG Engine" in data["app"]


@pytest.mark.asyncio
async def test_ingest_endpoint_rejects_non_pdf():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.txt", b"Texto no pdf", "text/plain")}
        response = await client.post("/api/v1/rag/ingest", files=files)
        assert response.status_code == 400
        assert "Solo se admiten archivos en formato PDF" in response.json()["detail"]


@pytest.mark.asyncio
async def test_query_endpoint_with_mocked_rag():
    mock_result = RAGResult(
        query="¿Cuál es el valor de pánico?",
        answer="El valor de pánico de glucosa es <45 o >450 mg/dL.",
        sources=[
            SourceDocumentInfo(
                source="PNO-001.pdf",
                page=1,
                snippet="Notificar en menos de 15 minutos.",
            )
        ],
    )

    with patch("app.api.v1.rag.generate_rag_response", new_callable=AsyncMock) as mock_rag:
        mock_rag.return_value = mock_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {"question": "¿Cuál es el valor de pánico?"}
            response = await client.post("/api/v1/rag/query", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == mock_result.answer
            assert len(data["sources"]) == 1
            assert data["sources"][0]["source"] == "PNO-001.pdf"
            assert data["sources"][0]["page"] == 1
