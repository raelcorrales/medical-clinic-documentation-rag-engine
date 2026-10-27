from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.v1.rag import router as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicialización de recursos al arrancar
    settings = get_settings()
    print(f"🚀 Iniciando {settings.APP_NAME} en modo {settings.APP_ENV}")
    print(f"📡 Vector Store: {settings.VECTOR_STORE_BACKEND} ({settings.ELASTICSEARCH_URL})")
    print(f"🧠 LLM: {settings.LLM_PROVIDER} ({settings.OLLAMA_BASE_URL} - {settings.OLLAMA_MODEL})")
    yield
    print("🛑 Deteniendo servicios...")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description="Motor RAG Asíncrono para Documentación Clínica Normativa (PNOs, ISO 15189, NOMs)",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Healthcheck
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "vector_store": settings.VECTOR_STORE_BACKEND,
            "llm_provider": settings.LLM_PROVIDER,
        }

    # API Routers
    app.include_router(rag_router, prefix="/api/v1")

    return app


app = create_app()
