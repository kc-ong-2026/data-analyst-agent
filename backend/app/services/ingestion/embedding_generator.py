"""Embedding generator for the folder-based RAG system."""

import logging
from typing import List

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import config
from app.db.models import (
    CATEGORY_MODELS,
    EmploymentDatasetMetadata,
    HoursWorkedDatasetMetadata,
    IncomeDatasetMetadata,
)
from app.services.llm_service import get_embedding_service

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings and tsvectors for folder-based dataset metadata."""

    # List of all folder metadata tables
    FOLDER_METADATA_MODELS = [
        EmploymentDatasetMetadata,
        HoursWorkedDatasetMetadata,
        IncomeDatasetMetadata,
    ]

    def __init__(self):
        self.batch_size = config.get_rag_config().get("embedding_batch_size", 100)

    async def update_embeddings(self, session: AsyncSession) -> dict:
        """Generate embeddings for all folder metadata rows missing them."""
        counts = {}
        total_count = 0

        embedding_service = get_embedding_service()

        # Process each folder metadata table
        for model_class in self.FOLDER_METADATA_MODELS:
            table_name = model_class.__tablename__
            category = table_name.replace("_dataset_metadata", "")

            # Get rows missing embeddings
            result = await session.execute(
                select(model_class).where(model_class.embedding.is_(None))
            )
            rows = result.scalars().all()

            if rows:
                texts = [row.summary_text or "" for row in rows]
                embeddings = await self._batch_embed(embedding_service, texts)

                for row, embedding in zip(rows, embeddings):
                    row.embedding = embedding

                await session.flush()
                counts[category] = len(rows)
                total_count += len(rows)
                logger.info(f"Generated {len(rows)} embeddings for {table_name}")

        # Update tsvectors
        await self._update_tsvectors(session)

        # Create vector indexes if enough data
        await self._create_vector_indexes(session)

        counts["total"] = total_count
        return counts

    async def _batch_embed(
        self, embedding_service, texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings in batches."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Filter empty texts
            batch = [t if t else "empty" for t in batch]
            embeddings = await embedding_service.embed_texts(batch)
            all_embeddings.extend(embeddings)
            logger.info(
                f"Embedded batch {i // self.batch_size + 1}"
                f"/{(len(texts) - 1) // self.batch_size + 1}"
            )

        return all_embeddings

    async def _update_tsvectors(self, session: AsyncSession) -> None:
        """Update tsvector columns for full-text search on all folder metadata tables."""
        for model_class in self.FOLDER_METADATA_MODELS:
            table_name = model_class.__tablename__

            await session.execute(
                text(
                    f"UPDATE {table_name} SET tsv = to_tsvector('english', "
                    f"COALESCE(summary_text, '') || ' ' || COALESCE(description, '') || ' ' || "
                    f"COALESCE(file_name, '')) "
                    f"WHERE tsv IS NULL"
                )
            )

        await session.flush()
        logger.info("Updated tsvector columns for all folder metadata tables")

    async def _create_vector_indexes(self, session: AsyncSession) -> None:
        """Create IVFFlat vector indexes if enough data exists in each folder metadata table."""
        for model_class in self.FOLDER_METADATA_MODELS:
            table_name = model_class.__tablename__
            index_name = f"idx_{table_name}_embedding"

            # Count rows
            result = await session.execute(text(f"SELECT count(*) FROM {table_name}"))
            row_count = result.scalar()

            if row_count and row_count >= 5:
                # Use sqrt(n) lists for IVFFlat, min 1
                lists = max(1, int(row_count ** 0.5))

                try:
                    # Drop existing index if any
                    await session.execute(text(f"DROP INDEX IF EXISTS {index_name}"))

                    # Create IVFFlat index
                    await session.execute(
                        text(
                            f"CREATE INDEX {index_name} ON {table_name} "
                            f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
                        )
                    )
                    logger.info(
                        f"Created IVFFlat index on {table_name} with {lists} lists"
                    )
                except Exception as e:
                    logger.warning(f"Could not create vector index on {table_name}: {e}")
