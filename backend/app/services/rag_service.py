"""RAG service with hybrid vector + full-text search and RRF fusion."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import config
from app.db.models import DataChunk, DatasetMetadata, RawDataTable
from app.db.session import get_db
from app.services.llm_service import get_embedding_service
from app.services.rag_models import ChunkResult, DatasetResult, RetrievalResult

logger = logging.getLogger(__name__)


class RAGService:
    """Hybrid retrieval engine using pgvector + full-text search with RRF fusion."""

    def __init__(self):
        rag_config = config.get_rag_config()
        self.vector_top_k = rag_config["vector_search_top_k"]
        self.fulltext_top_k = rag_config["fulltext_search_top_k"]
        self.hybrid_top_k = rag_config["hybrid_top_k"]
        self.rrf_k = rag_config["rrf_k"]
        self.similarity_threshold = rag_config["similarity_threshold"]

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
        year_filter: Optional[Dict[str, int]] = None,
    ) -> RetrievalResult:
        """Perform hybrid retrieval: vector search + full-text search + RRF fusion.

        Args:
            query: The search query.
            top_k: Number of results to return.
            category_filter: Optional category to filter by.
            year_filter: Optional year range dict {"start": int, "end": int}.

        Returns:
            RetrievalResult with chunks, datasets, and raw data.
        """
        top_k = top_k or self.hybrid_top_k

        embedding_service = get_embedding_service()
        query_embedding = await embedding_service.embed_query(query)

        async with get_db() as session:
            # Run vector search and full-text search
            vector_results = await self._vector_search(
                session, query_embedding, category_filter, year_filter
            )
            fulltext_results = await self._fulltext_search(
                session, query, category_filter, year_filter
            )

            # RRF fusion
            fused = self._rrf_fuse(vector_results, fulltext_results, top_k)

            # Fetch dataset metadata for matched chunks
            dataset_ids = list({c.dataset_id for c in fused})
            datasets = await self._get_datasets(session, dataset_ids)

            # Fetch raw data for matched chunks
            raw_data = await self._get_raw_data(session, fused, dataset_ids)

        return RetrievalResult(
            query=query,
            chunks=fused,
            datasets=datasets,
            raw_data=raw_data,
        )

    async def _vector_search(
        self,
        session: AsyncSession,
        query_embedding: List[float],
        category_filter: Optional[str],
        year_filter: Optional[Dict[str, int]],
    ) -> List[ChunkResult]:
        """Perform cosine similarity search on chunk embeddings."""
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        where_clauses = ["c.embedding IS NOT NULL"]
        params: Dict[str, Any] = {"embedding": embedding_str, "limit": self.vector_top_k}

        if category_filter:
            where_clauses.append("m.category = :category")
            params["category"] = category_filter

        if year_filter:
            if year_filter.get("start"):
                where_clauses.append(
                    "(m.year_range->>'max')::int >= :year_start"
                )
                params["year_start"] = year_filter["start"]
            if year_filter.get("end"):
                where_clauses.append(
                    "(m.year_range->>'min')::int <= :year_end"
                )
                params["year_end"] = year_filter["end"]

        where_sql = " AND ".join(where_clauses)

        sql = text(f"""
            SELECT c.id, c.dataset_id, m.dataset_name, c.chunk_type,
                   c.group_key, c.chunk_text, c.row_count, c.numeric_summary,
                   c.embedding <=> :embedding::vector AS distance
            FROM data_chunks c
            JOIN dataset_metadata m ON c.dataset_id = m.id
            WHERE {where_sql}
            ORDER BY distance
            LIMIT :limit
        """)

        result = await session.execute(sql, params)
        rows = result.fetchall()

        return [
            ChunkResult(
                chunk_id=row[0],
                dataset_id=row[1],
                dataset_name=row[2],
                chunk_type=row[3],
                group_key=row[4] or {},
                chunk_text=row[5],
                row_count=row[6],
                numeric_summary=row[7] or {},
                score=1.0 - (row[8] or 1.0),  # Convert distance to similarity
            )
            for row in rows
        ]

    async def _fulltext_search(
        self,
        session: AsyncSession,
        query: str,
        category_filter: Optional[str],
        year_filter: Optional[Dict[str, int]],
    ) -> List[ChunkResult]:
        """Perform full-text search using tsvector/tsquery."""
        where_clauses = ["c.tsv IS NOT NULL"]
        params: Dict[str, Any] = {"query": query, "limit": self.fulltext_top_k}

        if category_filter:
            where_clauses.append("m.category = :category")
            params["category"] = category_filter

        if year_filter:
            if year_filter.get("start"):
                where_clauses.append(
                    "(m.year_range->>'max')::int >= :year_start"
                )
                params["year_start"] = year_filter["start"]
            if year_filter.get("end"):
                where_clauses.append(
                    "(m.year_range->>'min')::int <= :year_end"
                )
                params["year_end"] = year_filter["end"]

        where_sql = " AND ".join(where_clauses)

        sql = text(f"""
            SELECT c.id, c.dataset_id, m.dataset_name, c.chunk_type,
                   c.group_key, c.chunk_text, c.row_count, c.numeric_summary,
                   ts_rank_cd(c.tsv, plainto_tsquery('english', :query)) AS rank
            FROM data_chunks c
            JOIN dataset_metadata m ON c.dataset_id = m.id
            WHERE {where_sql}
              AND c.tsv @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """)

        result = await session.execute(sql, params)
        rows = result.fetchall()

        return [
            ChunkResult(
                chunk_id=row[0],
                dataset_id=row[1],
                dataset_name=row[2],
                chunk_type=row[3],
                group_key=row[4] or {},
                chunk_text=row[5],
                row_count=row[6],
                numeric_summary=row[7] or {},
                score=float(row[8] or 0),
            )
            for row in rows
        ]

    def _rrf_fuse(
        self,
        vector_results: List[ChunkResult],
        fulltext_results: List[ChunkResult],
        top_k: int,
    ) -> List[ChunkResult]:
        """Reciprocal Rank Fusion to combine vector and full-text results."""
        scores: Dict[int, float] = defaultdict(float)
        chunks_by_id: Dict[int, ChunkResult] = {}

        # Score from vector search
        for rank, chunk in enumerate(vector_results):
            scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank + 1)
            chunks_by_id[chunk.chunk_id] = chunk

        # Score from full-text search
        for rank, chunk in enumerate(fulltext_results):
            scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank + 1)
            if chunk.chunk_id not in chunks_by_id:
                chunks_by_id[chunk.chunk_id] = chunk

        # Sort by combined RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for chunk_id in sorted_ids[:top_k]:
            chunk = chunks_by_id[chunk_id]
            chunk.score = scores[chunk_id]
            results.append(chunk)

        return results

    async def _get_datasets(
        self, session: AsyncSession, dataset_ids: List[int]
    ) -> List[DatasetResult]:
        """Fetch dataset metadata for given IDs."""
        if not dataset_ids:
            return []

        result = await session.execute(
            select(DatasetMetadata).where(DatasetMetadata.id.in_(dataset_ids))
        )
        rows = result.scalars().all()

        return [
            DatasetResult(
                dataset_id=row.id,
                dataset_name=row.dataset_name,
                file_path=row.file_path,
                category=row.category or "",
                description=row.description or "",
                columns=row.columns or [],
                row_count=row.row_count or 0,
                year_range=row.year_range or {},
                summary_text=row.summary_text or "",
            )
            for row in rows
        ]

    async def _get_raw_data(
        self,
        session: AsyncSession,
        chunks: List[ChunkResult],
        dataset_ids: List[int],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch raw data from raw tables using chunk group_keys."""
        if not chunks or not dataset_ids:
            return {}

        # Get raw table names for these datasets
        result = await session.execute(
            select(RawDataTable).where(RawDataTable.dataset_id.in_(dataset_ids))
        )
        raw_tables = {rt.dataset_id: rt for rt in result.scalars().all()}

        raw_data: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in chunks:
            rt = raw_tables.get(chunk.dataset_id)
            if not rt:
                continue

            # Build WHERE clause from group_key
            where_parts = []
            params: Dict[str, Any] = {}
            col_map = {c["original"]: c["safe"] for c in (rt.columns or [])}

            for key, value in (chunk.group_key or {}).items():
                if value is None:
                    continue
                # Find the safe column name
                safe_col = col_map.get(key, key)
                param_name = f"p_{safe_col}"
                where_parts.append(f'"{safe_col}"::text = :{param_name}')
                params[param_name] = str(value)

            where_sql = " AND ".join(where_parts) if where_parts else "1=1"

            try:
                sql = text(
                    f'SELECT * FROM "{rt.table_name}" WHERE {where_sql} LIMIT 100'
                )
                result = await session.execute(sql, params)
                rows = result.fetchall()
                columns = result.keys()

                key_label = f"{chunk.dataset_name}"
                if chunk.group_key:
                    key_parts = [f"{k}={v}" for k, v in chunk.group_key.items() if v]
                    if key_parts:
                        key_label += f" ({', '.join(key_parts)})"

                if key_label not in raw_data:
                    raw_data[key_label] = [
                        {col: val for col, val in zip(columns, row)}
                        for row in rows
                    ]
            except Exception as e:
                logger.warning(f"Failed to query raw table {rt.table_name}: {e}")

        return raw_data
