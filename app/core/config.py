from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    APP_NAME: str = "Medical RAG Engine"
    DEBUG: bool = True

    # Providers
    LLM_PROVIDER: Literal["ollama", "openai", "vllm", "openai_compatible"] = "ollama"
    EMBEDDING_PROVIDER: Literal["ollama", "openai", "vllm", "openai_compatible"] = "ollama"

    # Ollama (Local Development / Zero Token Cost)
    OLLAMA_BASE_URL: str = "http://192.168.3.4:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_TEMPERATURE: float = 0.0

    # OpenAI (Official Cloud API)
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

    # Custom / vLLM / OpenAI-Compatible (Production clusters, Triton, Groq, DeepSeek)
    CUSTOM_LLM_BASE_URL: str | None = None
    CUSTOM_LLM_API_KEY: str = "EMPTY"
    CUSTOM_LLM_MODEL: str = "llama-3.1-70b-instruct"
    CUSTOM_LLM_TEMPERATURE: float = 0.0

    CUSTOM_EMBED_BASE_URL: str | None = None
    CUSTOM_EMBED_API_KEY: str = "EMPTY"
    CUSTOM_EMBED_MODEL: str = "bge-m3"

    # Vector Store
    VECTOR_STORE_BACKEND: Literal["elasticsearch", "chroma"] = "elasticsearch"

    # Elasticsearch (Local Docker or Production Cluster)
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX: str = "clinical_docs"
    ELASTICSEARCH_API_KEY: str | None = None
    ELASTICSEARCH_USERNAME: str | None = None
    ELASTICSEARCH_PASSWORD: str | None = None
    ELASTICSEARCH_SSL_VERIFY: bool = False

    # RAG parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
