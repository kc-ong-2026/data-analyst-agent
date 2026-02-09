"""Embedding generator for the RAG system."""

import logging
from typing import List

from sqlalchemy import text, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import config
from app.db.models import DataChunk, DatasetMetadata
from app.services.llm_service import get_embedding_service

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings and tsvectors for dataset metadata and chunks."""

    def __init__(self):
        self.batch_size = config.get_rag_config().get("embedding_batch_size", 100)

    async def update_embeddings(self, session: AsyncSession) -> dict:
        """Generate embeddings for all rows missing them."""
        counts = {"metadata": 0, "chunks": 0}

        embedding_service = get_embedding_service()

        # Process dataset metadata
        result = await session.execute(
            select(DatasetMetadata).where(DatasetMetadata.embedding.is_(None))
        )
        metadata_rows = result.scalars().all()

        if metadata_rows:
            texts = [row.summary_text or "" for row in metadata_rows]
            embeddings = await self._batch_embed(embedding_service, texts)

            for row, embedding in zip(metadata_rows, embeddings):
                row.embedding = embedding
                counts["metadata"] += 1

            await session.flush()
            logger.info(f"Generated {counts['metadata']} metadata embeddings")

        # Process data chunks
        result = await session.execute(
            select(DataChunk).where(DataChunk.embedding.is_(None))
        )
        chunk_rows = result.scalars().all()

        if chunk_rows:
            texts = [row.chunk_text or "" for row in chunk_rows]
            embeddings = await self._batch_embed(embedding_service, texts)

            for row, embedding in zip(chunk_rows, embeddings):
                row.embedding = embedding
                counts["chunks"] += 1

            await session.flush()
            logger.info(f"Generated {counts['chunks']} chunk embeddings")

        # Update tsvectors
        await self._update_tsvectors(session)

        # Create vector indexes if enough data
        await self._create_vector_indexes(session)

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
        """Update tsvector columns for full-text search."""
        await session.execute(
            text(
                "UPDATE dataset_metadata SET tsv = to_tsvector('english', "
                "COALESCE(summary_text, '') || ' ' || COALESCE(description, '') || ' ' || "
                "COALESCE(dataset_name, '') || ' ' || COALESCE(category, '')) "
                "WHERE tsv IS NULL"
            )
        )
        await session.execute(
            text(
                "UPDATE data_chunks SET tsv = to_tsvector('english', "
                "COALESCE(chunk_text, '')) WHERE tsv IS NULL"
            )
        )
        await session.flush()
        logger.info("Updated tsvector columns")

    async def _create_vector_indexes(self, session: AsyncSession) -> None:
        """Create IVFFlat vector indexes if enough data exists."""
        # IVFFlat needs at least lists * rows to work; use sqrt(n) lists
        result = await session.execute(text("SELECT count(*) FROM data_chunks"))
        chunk_count = result.scalar()

        if chunk_count and chunk_count >= 100:
            lists = max(1, int(chunk_count ** 0.5))
            try:
                await session.execute(
                    text("DROP INDEX IF EXISTS idx_data_chunks_embedding")
                )
                await session.execute(
                    text(
                        f"CREATE INDEX idx_data_chunks_embedding ON data_chunks "
                        f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
                    )
                )
                logger.info(
                    f"Created IVFFlat index on data_chunks with {lists} lists"
                )
            except Exception as e:
                logger.warning(f"Could not create vector index on chunks: {e}")

        result = await session.execute(text("SELECT count(*) FROM dataset_metadata"))
        meta_count = result.scalar()

        if meta_count and meta_count >= 10:
            lists = max(1, int(meta_count ** 0.5))
            try:
                await session.execute(
                    text("DROP INDEX IF EXISTS idx_dataset_metadata_embedding")
                )
                await session.execute(
                    text(
                        f"CREATE INDEX idx_dataset_metadata_embedding ON dataset_metadata "
                        f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
                    )
                )
                logger.info(
                    f"Created IVFFlat index on dataset_metadata with {lists} lists"
                )
            except Exception as e:
                logger.warning(f"Could not create vector index on metadata: {e}")
