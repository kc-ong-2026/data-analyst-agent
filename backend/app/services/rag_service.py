"""RAG service with folder-based hybrid vector + BM25 search, RRF fusion, and cross-encoder reranking."""

import asyncio
import logging
import pickle
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import config
from app.db.models import (
    EmploymentDatasetMetadata,
    HoursWorkedDatasetMetadata,
    IncomeDatasetMetadata,
)
from app.db.session import get_db
from app.services.llm_service import get_embedding_service
from app.services.rag_models import FolderRetrievalResult, MetadataResult, TableSchema

logger = logging.getLogger(__name__)


class RAGService:
    """Hybrid retrieval engine using folder-based metadata with pgvector + full-text search with RRF fusion."""

    # List of (category_name, model_class, table_name) tuples
    FOLDER_TABLES: list[tuple[str, type, str]] = [
        ("employment", EmploymentDatasetMetadata, "employment_dataset_metadata"),
        ("hours_worked", HoursWorkedDatasetMetadata, "hours_worked_dataset_metadata"),
        ("income", IncomeDatasetMetadata, "income_dataset_metadata"),
    ]

    def __init__(self):
        rag_config = config.get_rag_config()
        self.vector_top_k = rag_config["vector_search_top_k"]
        self.fulltext_top_k = rag_config["fulltext_search_top_k"]
        self.hybrid_top_k = rag_config["hybrid_top_k"]
        self.rrf_k = rag_config["rrf_k"]
        self.similarity_threshold = rag_config["similarity_threshold"]
        self.use_reranking = rag_config.get("use_reranking", True)
        self.use_bm25 = rag_config.get("use_bm25", True)
        self._bm25_cache = {}  # In-memory cache: {cache_key: (BM25Okapi, rows)}

        # BM25 disk cache directory
        self._bm25_cache_dir = Path("./data/bm25_cache")
        self._bm25_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[BM25 CACHE] Cache directory: {self._bm25_cache_dir}")

        # Thread pool for CPU-bound operations (reranking)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-rerank")

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category_filter: str | None = None,
        year_filter: dict[str, int] | None = None,
        use_reranking: bool | None = None,
        query_embedding: list[float] | None = None,
    ) -> FolderRetrievalResult:
        """Perform hybrid retrieval across folder metadata tables with optional reranking.

        Args:
            query: The search query.
            top_k: Number of results to return.
            category_filter: Optional category to filter by (employment, hours_worked, income).
            year_filter: Optional year range dict {"start": int, "end": int}.
            use_reranking: Override config setting for reranking (default: use config value).
            query_embedding: Pre-computed query embedding (optional, will compute if not provided).

        Returns:
            FolderRetrievalResult with metadata and table schemas.
        """
        top_k = top_k or self.hybrid_top_k
        use_reranking = use_reranking if use_reranking is not None else self.use_reranking

        # Use pre-computed embedding if provided, otherwise compute it
        if query_embedding is None:
            embedding_service = get_embedding_service()
            query_embedding = await embedding_service.embed_query(query)
        else:
            logger.info("[PERF] Using pre-computed query embedding, skipping re-embedding")

        async with get_db() as session:
            all_metadata_results: list[MetadataResult] = []

            # Search each folder table (or just the filtered category)
            folders_to_search = self.FOLDER_TABLES
            if category_filter:
                logger.info(f"[RAG FILTER] Applying category filter: {category_filter}")
                folders_to_search = [
                    (cat, model, table)
                    for cat, model, table in self.FOLDER_TABLES
                    if cat == category_filter
                ]
                logger.info(
                    f"[RAG FILTER] Narrowed search from {len(self.FOLDER_TABLES)} tables to {len(folders_to_search)} table(s)"
                )
            else:
                logger.info(
                    f"[RAG FILTER] No category filter - searching all {len(self.FOLDER_TABLES)} tables"
                )

            for category, model_class, table_name in folders_to_search:
                logger.info(f"Searching folder table: {category} ({table_name})")
                results = await self._search_folder_table(
                    session,
                    model_class,
                    table_name,
                    category,
                    query,
                    query_embedding,
                    year_filter,
                )
                logger.info(f"  -> Found {len(results)} results in {category}")
                all_metadata_results.extend(results)

            # Sort by score and take top_k
            all_metadata_results.sort(key=lambda x: x.score, reverse=True)
            top_metadata_results = all_metadata_results[:top_k]

            # Apply reranking if enabled
            if use_reranking and top_metadata_results:
                logger.info(
                    f"Applying cross-encoder reranking to {len(top_metadata_results)} results"
                )
                top_metadata_results = await self._rerank_results(query, top_metadata_results)

            # Get full table schemas for SQL generation
            table_schemas = await self._get_table_schemas(session, top_metadata_results)

        return FolderRetrievalResult(
            query=query,
            metadata_results=top_metadata_results,
            table_schemas=table_schemas,
            total_results=len(all_metadata_results),
        )

    async def _search_folder_table(
        self,
        session: AsyncSession,
        model_class: type,
        table_name: str,
        category: str,
        query: str,
        query_embedding: list[float],
        year_filter: dict[str, int] | None,
    ) -> list[MetadataResult]:
        """Perform hybrid search (vector + BM25/full-text + RRF) on a single folder metadata table.

        Args:
            session: Database session
            model_class: ORM model class for this folder
            table_name: SQL table name
            category: Category name
            query: Search query
            query_embedding: Query embedding vector
            year_filter: Optional year range filter

        Returns:
            List of MetadataResult objects with RRF scores
        """
        # Run vector search
        vector_results = await self._folder_vector_search(
            session, table_name, category, query_embedding, year_filter
        )

        # Run BM25 or full-text search based on config
        if self.use_bm25:
            text_results = await self._folder_bm25_search(
                session, table_name, category, query, year_filter
            )
        else:
            text_results = await self._folder_fulltext_search(
                session, table_name, category, query, year_filter
            )

        # RRF fusion
        fused = self._rrf_fuse_metadata(vector_results, text_results, top_k=10)

        return fused

    async def _folder_vector_search(
        self,
        session: AsyncSession,
        table_name: str,
        category: str,
        query_embedding: list[float],
        year_filter: dict[str, int] | None,
    ) -> list[MetadataResult]:
        """Perform cosine similarity search on folder metadata embeddings."""
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        where_clauses = ["m.embedding IS NOT NULL"]
        params: dict[str, Any] = {"limit": self.vector_top_k}

        if year_filter:
            if year_filter.get("start"):
                where_clauses.append("(m.year_range->>'max')::int >= :year_start")
                params["year_start"] = year_filter["start"]
            if year_filter.get("end"):
                where_clauses.append("(m.year_range->>'min')::int <= :year_end")
                params["year_end"] = year_filter["end"]

        where_sql = " AND ".join(where_clauses)

        sql = text(
            f"""
            SELECT m.id, m.file_name, m.file_path, m.table_name, m.description,
                   m.columns, m.primary_dimensions, m.numeric_columns, m.categorical_columns,
                   m.row_count, m.year_range, m.summary_text,
                   m.embedding <=> '{embedding_str}'::vector AS distance
            FROM {table_name} m
            WHERE {where_sql}
            ORDER BY distance
            LIMIT :limit
        """
        )

        result = await session.execute(sql, params)
        rows = result.fetchall()

        return [
            MetadataResult(
                metadata_id=row[0],
                category=category,
                file_name=row[1],
                file_path=row[2],
                table_name=row[3],
                description=row[4] or "",
                columns=row[5] or [],
                primary_dimensions=row[6] or [],
                numeric_columns=row[7] or [],
                categorical_columns=row[8] or [],
                row_count=row[9] or 0,
                year_range=row[10],
                summary_text=row[11] or "",
                score=1.0 - (row[12] or 1.0),  # Convert distance to similarity
            )
            for row in rows
        ]

    async def _folder_fulltext_search(
        self,
        session: AsyncSession,
        table_name: str,
        category: str,
        query: str,
        year_filter: dict[str, int] | None,
    ) -> list[MetadataResult]:
        """Perform full-text search using tsvector/tsquery on folder metadata."""
        where_clauses = ["m.tsv IS NOT NULL"]
        params: dict[str, Any] = {"query": query, "limit": self.fulltext_top_k}

        if year_filter:
            if year_filter.get("start"):
                where_clauses.append("(m.year_range->>'max')::int >= :year_start")
                params["year_start"] = year_filter["start"]
            if year_filter.get("end"):
                where_clauses.append("(m.year_range->>'min')::int <= :year_end")
                params["year_end"] = year_filter["end"]

        where_sql = " AND ".join(where_clauses)

        sql = text(
            f"""
            SELECT m.id, m.file_name, m.file_path, m.table_name, m.description,
                   m.columns, m.primary_dimensions, m.numeric_columns, m.categorical_columns,
                   m.row_count, m.year_range, m.summary_text,
                   ts_rank_cd(m.tsv, plainto_tsquery('english', :query)) AS rank
            FROM {table_name} m
            WHERE {where_sql}
              AND m.tsv @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """
        )

        result = await session.execute(sql, params)
        rows = result.fetchall()

        return [
            MetadataResult(
                metadata_id=row[0],
                category=category,
                file_name=row[1],
                file_path=row[2],
                table_name=row[3],
                description=row[4] or "",
                columns=row[5] or [],
                primary_dimensions=row[6] or [],
                numeric_columns=row[7] or [],
                categorical_columns=row[8] or [],
                row_count=row[9] or 0,
                year_range=row[10],
                summary_text=row[11] or "",
                score=float(row[12] or 0),
            )
            for row in rows
        ]

    async def _folder_bm25_search(
        self,
        session: AsyncSession,
        table_name: str,
        category: str,
        query: str,
        year_filter: dict[str, int] | None,
    ) -> list[MetadataResult]:
        """Perform BM25 search on folder metadata with disk persistence.

        Replaces PostgreSQL full-text search with proper BM25 ranking.
        Caches BM25 indexes to disk for faster startup.
        """
        # Build cache key
        cache_key = f"{category}_{table_name}"
        cache_file = self._bm25_cache_dir / f"{cache_key}.pkl"

        # Try to load from in-memory cache first
        if cache_key in self._bm25_cache:
            logger.debug(f"[BM25 CACHE] In-memory HIT: {cache_key}")
            bm25, rows = self._bm25_cache[cache_key]
        # Try to load from disk if not in memory
        elif cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    bm25, rows = pickle.load(f)
                self._bm25_cache[cache_key] = (bm25, rows)
                logger.info(f"[BM25 CACHE] Loaded from disk: {cache_key} ({len(rows)} docs)")
            except Exception as e:
                logger.warning(f"[BM25 CACHE] Failed to load from disk: {cache_key}, error: {e}")
                bm25, rows = None, None
        else:
            bm25, rows = None, None

        # If not cached, build new index
        if bm25 is None or rows is None:
            # Fetch metadata documents (with LIMIT to avoid loading entire table)
            where_clauses = ["m.summary_text IS NOT NULL"]
            params = {"limit": 1000}  # Limit corpus size for performance

            if year_filter:
                if year_filter.get("start"):
                    where_clauses.append("(m.year_range->>'max')::int >= :year_start")
                    params["year_start"] = year_filter["start"]
                if year_filter.get("end"):
                    where_clauses.append("(m.year_range->>'min')::int <= :year_end")
                    params["year_end"] = year_filter["end"]

            where_sql = " AND ".join(where_clauses)

            sql = text(
                f"""
                SELECT m.id, m.file_name, m.file_path, m.table_name, m.description,
                       m.columns, m.primary_dimensions, m.numeric_columns, m.categorical_columns,
                       m.row_count, m.year_range, m.summary_text
                FROM {table_name} m
                WHERE {where_sql}
                LIMIT :limit
            """
            )

            result = await session.execute(sql, params)
            rows = result.fetchall()

            if not rows:
                return []

            # Build BM25 index
            logger.info(f"[BM25 CACHE] Building index for {cache_key} ({len(rows)} docs)")
            corpus = [self._tokenize(row[11]) for row in rows]  # summary_text
            bm25 = BM25Okapi(corpus)

            # Cache in memory
            self._bm25_cache[cache_key] = (bm25, rows)

            # Persist to disk
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump((bm25, rows), f)
                logger.info(f"[BM25 CACHE] Persisted to disk: {cache_key}")
            except Exception as e:
                logger.warning(f"[BM25 CACHE] Failed to persist to disk: {cache_key}, error: {e}")

        # Score documents with BM25
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)

        # Get top results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            : self.fulltext_top_k
        ]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:  # Skip zero scores
                continue
            row = rows[idx]
            results.append(
                MetadataResult(
                    metadata_id=row[0],
                    category=category,
                    file_name=row[1],
                    file_path=row[2],
                    table_name=row[3],
                    description=row[4] or "",
                    columns=row[5] or [],
                    primary_dimensions=row[6] or [],
                    numeric_columns=row[7] or [],
                    categorical_columns=row[8] or [],
                    row_count=row[9] or 0,
                    year_range=row[10],
                    summary_text=row[11] or "",
                    score=float(scores[idx]),  # BM25 score
                )
            )

        return results

    async def warmup_caches(self) -> None:
        """Pre-load BM25 indexes from disk on startup.

        This method should be called during application startup to pre-warm
        the BM25 cache, avoiding cold-start latency on first request.
        """
        logger.info("[BM25 CACHE] Warming up BM25 caches from disk...")
        loaded_count = 0

        for cache_file in self._bm25_cache_dir.glob("*.pkl"):
            cache_key = cache_file.stem
            try:
                with open(cache_file, "rb") as f:
                    bm25, rows = pickle.load(f)
                self._bm25_cache[cache_key] = (bm25, rows)
                loaded_count += 1
                logger.info(f"[BM25 CACHE] Loaded {cache_key} ({len(rows)} docs)")
            except Exception as e:
                logger.warning(f"[BM25 CACHE] Failed to load {cache_key}: {e}")

        logger.info(f"[BM25 CACHE] Warmup complete: {loaded_count} indexes loaded")

    def clear_bm25_cache(self) -> None:
        """Clear both in-memory and disk BM25 caches.

        Call this after data ingestion to ensure fresh indexes.
        """
        logger.info("[BM25 CACHE] Clearing BM25 caches (memory + disk)")

        # Clear in-memory cache
        self._bm25_cache.clear()

        # Clear disk cache
        for cache_file in self._bm25_cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                logger.debug(f"[BM25 CACHE] Deleted {cache_file.name}")
            except Exception as e:
                logger.warning(f"[BM25 CACHE] Failed to delete {cache_file.name}: {e}")

        logger.info("[BM25 CACHE] Cache cleared")

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 (simple whitespace + lowercase)."""
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    async def _rerank_results(
        self,
        query: str,
        metadata_results: list[MetadataResult],
    ) -> list[MetadataResult]:
        """Rerank metadata results using cross-encoder (async, non-blocking).

        Args:
            query: User query
            metadata_results: Initial results from hybrid search

        Returns:
            Reranked results with updated scores
        """
        if not metadata_results:
            return metadata_results

        # Build documents for reranking (use summary_text + description)
        documents = []
        for result in metadata_results:
            doc = f"{result.description}\n{result.summary_text}"
            documents.append(doc)

        # Offload CPU-bound reranking to thread pool (non-blocking)
        from app.services.reranker import get_reranker

        reranker = get_reranker()

        loop = asyncio.get_event_loop()
        logger.debug(f"[RERANK] Starting async reranking for {len(documents)} documents")

        # Run reranking in thread pool to avoid blocking event loop
        ranked_indices_scores = await loop.run_in_executor(
            self._executor,
            reranker.rerank,
            query,
            documents,
            len(documents),  # Rerank all
        )

        # Reorder results and update scores
        reranked_results = []
        for idx, rerank_score in ranked_indices_scores:
            result = metadata_results[idx]
            # Update score with reranker output
            result.score = rerank_score
            reranked_results.append(result)

        # Sort by score descending (highest relevance first)
        reranked_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            f"[RERANK] Completed async reranking: {len(reranked_results)} results. "
            f"Top score: {reranked_results[0].score:.3f}, "
            f"Bottom score: {reranked_results[-1].score:.3f}"
        )

        return reranked_results

    def _rrf_fuse_metadata(
        self,
        vector_results: list[MetadataResult],
        fulltext_results: list[MetadataResult],
        top_k: int,
    ) -> list[MetadataResult]:
        """Reciprocal Rank Fusion to combine vector and full-text metadata results."""
        scores: dict[int, float] = defaultdict(float)
        metadata_by_id: dict[int, MetadataResult] = {}

        # Score from vector search
        for rank, metadata in enumerate(vector_results):
            scores[metadata.metadata_id] += 1.0 / (self.rrf_k + rank + 1)
            metadata_by_id[metadata.metadata_id] = metadata

        # Score from full-text search
        for rank, metadata in enumerate(fulltext_results):
            scores[metadata.metadata_id] += 1.0 / (self.rrf_k + rank + 1)
            if metadata.metadata_id not in metadata_by_id:
                metadata_by_id[metadata.metadata_id] = metadata

        # Sort by combined RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for metadata_id in sorted_ids[:top_k]:
            metadata = metadata_by_id[metadata_id]
            metadata.score = scores[metadata_id]
            results.append(metadata)

        return results

    async def _get_table_schemas(
        self,
        session: AsyncSession,
        metadata_results: list[MetadataResult],
    ) -> list[TableSchema]:
        """Fetch full table schemas for SQL generation.

        Args:
            session: Database session
            metadata_results: List of metadata results

        Returns:
            List of TableSchema objects with formatted SQL schema prompts and scores
        """
        table_schemas = []

        # Create score lookup by metadata_id
        scores_by_id = {r.metadata_id: r.score for r in metadata_results}

        # Group by category to minimize queries
        by_category: dict[str, list[MetadataResult]] = defaultdict(list)
        for result in metadata_results:
            by_category[result.category].append(result)

        # Fetch schemas from each folder table
        for category, results in by_category.items():
            # Find the model class for this category
            model_class = None
            for cat, model, table in self.FOLDER_TABLES:
                if cat == category:
                    model_class = model
                    break

            if not model_class:
                continue

            # Get metadata IDs
            metadata_ids = [r.metadata_id for r in results]

            # Fetch full metadata objects
            db_result = await session.execute(
                select(model_class).where(model_class.id.in_(metadata_ids))
            )
            metadata_rows = db_result.scalars().all()

            # Convert to TableSchema objects
            for row in metadata_rows:
                table_schemas.append(
                    TableSchema(
                        table_name=row.table_name,
                        category=category,
                        description=row.description or "",
                        columns=row.columns or [],
                        primary_dimensions=row.primary_dimensions or [],
                        numeric_columns=row.numeric_columns or [],
                        categorical_columns=row.categorical_columns or [],
                        row_count=row.row_count or 0,
                        year_range=row.year_range,
                        sql_schema_prompt=row.get_sql_schema_prompt(),
                        summary_text=getattr(row, "summary_text", "")
                        or "",  # Rich context about dataset
                        file_path=getattr(
                            row, "file_path", None
                        ),  # Include file path for DataFrame loading
                        score=scores_by_id.get(row.id, 0.0),  # Include relevance score
                    )
                )

        # Sort by score descending to preserve reranked order
        table_schemas.sort(key=lambda x: x.score, reverse=True)

        return table_schemas
