from typing import Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.services.llm_factory import get_llm
from app.services.vector_store import get_retriever


class SourceDocumentInfo(BaseModel):
    """Metadatos de trazabilidad de los fragmentos fuente utilizados."""
    source: str = Field(description="Nombre del archivo fuente")
    page: int = Field(description="Número de página del documento")
    snippet: str = Field(description="Extracto del fragmento de texto recuperado")


class RAGResult(BaseModel):
    """Resultado de la consulta RAG con respuesta generada y fuentes trazables."""
    query: str
    answer: str
    sources: list[SourceDocumentInfo]


CLINICAL_RAG_PROMPT = """Eres un asistente técnico-clínico de laboratorio y hospital especializado en resolución operativa, códigos de error de analizadores, contingencias de bioseguridad (ISO 15189 / NOM-007) y escalamiento legal (NOM-004).

INSTRUCCIONES CLÍNICAS Y OPERATIVAS:
1. Responde a la pregunta basándote ÚNICA Y EXCLUSIVAMENTE en el contexto documental proporcionado.
2. Si la información solicitada no está explícitamente en el contexto, responde textualmente:
   "La información solicitada no se encuentra disponible en la documentación provista."
3. Estructura tu respuesta de forma clara y directa con los siguientes puntos (siempre que apliquen según el contexto):
   - **Acción Inmediata Requerida:** Instrucciones concretas, secuenciales y operativas.
   - **Criterio Técnico / Normativo / Código de Error:** Tiempos límite, parámetros de seguridad, reactivos o normas aplicables.
   - **Registro y Trazabilidad Obligatoria:** Formatos, bitácoras o firmas requeridas.
4. NO supongas, NO inventes códigos ni acciones que no estén en el texto. Sé quirúrgico y preciso.

Contexto clínico:
{context}

Pregunta:
{question}

Respuesta:"""


def _format_context(docs: list[Document]) -> str:
    """Formatea los documentos recuperados en un bloque de contexto legible para el LLM."""
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "desconocido")
        page = doc.metadata.get("page", 1)
        formatted_chunks.append(
            f"--- Documento: {source} (Página {page}) ---\n{doc.page_content}"
        )
    return "\n\n".join(formatted_chunks)


async def generate_rag_response(question: str, top_k: int | None = None) -> RAGResult:
    """Ejecuta la cadena RAG completa de recuperación y generación asíncrona."""
    retriever = get_retriever({"k": top_k} if top_k else None)
    
    # 1. Recuperación de fragmentos relevantes
    retrieved_docs: list[Document] = await retriever.ainvoke(question)

    if not retrieved_docs:
        return RAGResult(
            query=question,
            answer="La información solicitada no se encuentra disponible en la documentación provista.",
            sources=[],
        )

    context_str = _format_context(retrieved_docs)

    # 2. Generación con LLM vía LCEL
    prompt = ChatPromptTemplate.from_template(CLINICAL_RAG_PROMPT)
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    response_text = await chain.ainvoke({
        "context": context_str,
        "question": question,
    })

    # 3. Empaquetar fuentes con metadatos para trazabilidad clínica
    sources = [
        SourceDocumentInfo(
            source=doc.metadata.get("source", "desconocido"),
            page=doc.metadata.get("page", 1),
            snippet=doc.page_content[:250] + ("..." if len(doc.page_content) > 250 else ""),
        )
        for doc in retrieved_docs
    ]

    return RAGResult(
        query=question,
        answer=response_text.strip(),
        sources=sources,
    )
