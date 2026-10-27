from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from app.core.config import get_settings


def get_embeddings() -> Embeddings:
    """Retorna la instancia de Embeddings configurada (Ollama o OpenAI)."""
    settings = get_settings()

    if settings.EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBED_MODEL,
        )
    elif settings.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBED_MODEL,
        )
    elif settings.EMBEDDING_PROVIDER in ("vllm", "openai_compatible"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            base_url=settings.CUSTOM_EMBED_BASE_URL,
            api_key=settings.CUSTOM_EMBED_API_KEY,
            model=settings.CUSTOM_EMBED_MODEL,
            check_embedding_ctx_length=False,
        )
    else:
        raise ValueError(f"Proveedor de embeddings no soportado: {settings.EMBEDDING_PROVIDER}")


def get_llm() -> BaseChatModel:
    """Retorna la instancia del modelo de lenguaje configurado (Ollama, OpenAI o vLLM)."""
    settings = get_settings()

    if settings.LLM_PROVIDER == "ollama":
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=settings.OLLAMA_TEMPERATURE,
        )
    elif settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=0.0,
        )
    elif settings.LLM_PROVIDER in ("vllm", "openai_compatible"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=settings.CUSTOM_LLM_BASE_URL,
            api_key=settings.CUSTOM_LLM_API_KEY,
            model=settings.CUSTOM_LLM_MODEL,
            temperature=settings.CUSTOM_LLM_TEMPERATURE,
        )
    else:
        raise ValueError(f"Proveedor de LLM no soportado: {settings.LLM_PROVIDER}")
