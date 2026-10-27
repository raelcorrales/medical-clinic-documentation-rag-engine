from typing import Any
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_elasticsearch import ElasticsearchStore
from app.core.config import get_settings
from app.services.llm_factory import get_embeddings


def get_vector_store() -> VectorStore:
    """Instancia y retorna la base vectorial configurada (Elasticsearch o Chroma)."""
    settings = get_settings()
    embeddings = get_embeddings()

    if settings.VECTOR_STORE_BACKEND == "elasticsearch":
        es_params: dict[str, Any] = {
            "es_url": settings.ELASTICSEARCH_URL,
            "index_name": settings.ELASTICSEARCH_INDEX,
            "embedding": embeddings,
            "strategy": ElasticsearchStore.ApproxRetrievalStrategy(),
        }
        if settings.ELASTICSEARCH_API_KEY:
            es_params["es_api_key"] = settings.ELASTICSEARCH_API_KEY
        elif settings.ELASTICSEARCH_USERNAME and settings.ELASTICSEARCH_PASSWORD:
            es_params["es_user"] = settings.ELASTICSEARCH_USERNAME
            es_params["es_password"] = settings.ELASTICSEARCH_PASSWORD

        return ElasticsearchStore(**es_params)
    elif settings.VECTOR_STORE_BACKEND == "chroma":
        from langchain_community.vectorstores import Chroma

        return Chroma(
            collection_name=settings.ELASTICSEARCH_INDEX,
            embedding_function=embeddings,
            persist_directory="./chroma_data",
        )
    else:
        raise ValueError(f"Backend vectorial no soportado: {settings.VECTOR_STORE_BACKEND}")


def embed_and_index_chunks(chunks: list[Document]) -> int:
    """Indexa una lista de chunks con sus embeddings en la base vectorial."""
    if not chunks:
        return 0

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    return len(chunks)


def search_similar_chunks(query: str, k: int | None = None) -> list[Document]:
    """Busca los fragmentos más relevantes para una consulta dada."""
    settings = get_settings()
    top_k = k or settings.TOP_K
    vector_store = get_vector_store()
    return vector_store.similarity_search(query=query, k=top_k)


def get_retriever(search_kwargs: dict[str, Any] | None = None):
    """Retorna un Retriever de LangChain para integrar en la cadena RAG."""
    settings = get_settings()
    vector_store = get_vector_store()
    kwargs = search_kwargs or {"k": settings.TOP_K}
    return vector_store.as_retriever(search_kwargs=kwargs)
