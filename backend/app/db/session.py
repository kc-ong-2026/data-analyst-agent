"""Async SQLAlchemy engine and session factory."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import config

logger = logging.getLogger(__name__)

engine = None
async_session_factory: Optional[async_sessionmaker] = None


def init_db() -> None:
    """Initialize the async database engine and session factory."""
    global engine, async_session_factory

    db_config = config.get_database_config()
    database_url = db_config["url"]

    if not database_url:
        logger.warning("DATABASE_URL not set, database features disabled")
        return

    engine = create_async_engine(
        database_url,
        pool_size=db_config["pool_size"],
        max_overflow=db_config["max_overflow"],
        echo=db_config["echo"],
    )

    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    logger.info("Database engine initialized")


async def close_db() -> None:
    """Dispose the database engine."""
    global engine, async_session_factory
    if engine:
        await engine.dispose()
        engine = None
        async_session_factory = None
        logger.info("Database engine disposed")


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session."""
    if async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
