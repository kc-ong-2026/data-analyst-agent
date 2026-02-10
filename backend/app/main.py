"""Main FastAPI application entry point."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import config
from app.routes import chat, config as config_routes, data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set specific loggers to appropriate levels
logging.getLogger("app").setLevel(logging.INFO)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langgraph").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def _check_and_ingest() -> None:
    """Check if datasets are ingested; if not, run ingestion."""
    try:
        from app.db.session import async_session_factory, get_db
        if async_session_factory is None:
            logger.info("Database not configured, skipping ingestion")
            return

        from sqlalchemy import text

        async with get_db() as session:
            # Check if employment metadata table exists and has data
            try:
                result = await session.execute(text("SELECT count(*) FROM employment_dataset_metadata"))
                count = result.scalar()

                if count and count > 0:
                    logger.info(f"Database already has {count} employment datasets, skipping ingestion")
                    return
            except Exception:
                # Table doesn't exist yet, proceed with ingestion
                pass

        logger.info("No datasets found in database, starting ingestion...")

        from app.services.ingestion.data_processor import DataProcessor
        from app.services.ingestion.embedding_generator import EmbeddingGenerator

        async with get_db() as session:
            processor = DataProcessor()
            counts = await processor.process_all_datasets(session)
            logger.info(
                f"Ingested {counts['metadata_entries']} metadata entries, "
                f"{counts['data_tables']} data tables, "
                f"{counts['total_rows']} total rows"
            )

            # Temporarily disable embeddings due to OpenAI quota
            logger.info("Skipping embedding generation (OpenAI quota exceeded)")
            # generator = EmbeddingGenerator()
            # embed_counts = await generator.update_embeddings(session)
            # logger.info(
            #     f"Generated {embed_counts.get('total', 0)} embeddings across all categories"
            # )

    except Exception as e:
        logger.warning(f"Auto-ingestion check failed: {e}", exc_info=False)


async def _cleanup_expired_checkpoints():
    """Background task to clean up expired checkpoints every 10 minutes."""
    from app.services.checkpoint_service import get_checkpoint_service

    while True:
        await asyncio.sleep(600)  # 10 minutes
        checkpoint_service = get_checkpoint_service()
        checkpoint_service.cleanup_expired()
        logger.debug("Cleaned up expired checkpoints")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    # Startup
    print("Starting Govtech Chat Assistant...")

    # Initialize database
    from app.db.session import init_db, close_db
    init_db()

    # Auto-ingest if database is empty
    await _check_and_ingest()

    # Log LangSmith tracing status
    langsmith_config = config.get_langsmith_config()
    if langsmith_config["enabled"] and langsmith_config["api_key_configured"]:
        logger.info(
            f"LangSmith tracing ENABLED - Project: {langsmith_config['project_name']}"
        )
        logger.info(
            "View traces at: https://smith.langchain.com"
        )
    else:
        logger.info("LangSmith tracing DISABLED")
        if not langsmith_config["api_key_configured"]:
            logger.info("  Reason: LANGCHAIN_API_KEY not set in environment")
        elif not langsmith_config["enabled"]:
            logger.info("  Reason: langsmith.enabled=false in config.yaml")

    # Start checkpoint cleanup background task
    cleanup_task = asyncio.create_task(_cleanup_expired_checkpoints())
    logger.info("Started checkpoint cleanup background task")

    yield

    # Shutdown
    print("Shutting down Govtech Chat Assistant...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app_config = config.yaml_config.get("app", {})

    app = FastAPI(
        title=app_config.get("name", "Govtech Chat Assistant"),
        version=app_config.get("version", "1.0.0"),
        description="AI-powered chat assistant for Singapore government data analysis",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat.router, prefix="/api")
    app.include_router(config_routes.router, prefix="/api")
    app.include_router(data.router, prefix="/api")

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": app_config.get("name", "Govtech Chat Assistant"),
            "version": app_config.get("version", "1.0.0"),
            "docs": "/docs",
        }

    @app.get("/health")
    async def health():
        """Simple health check."""
        return {"status": "healthy"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.settings.host,
        port=config.settings.port,
        reload=config.settings.debug,
    )
