"""RAG service with folder-based hybrid vector + full-text search and RRF fusion."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import config
from app.db.models import (
    EmploymentDatasetMetadata,
    HoursWorkedDatasetMetadata,
    IncomeDatasetMetadata,
    LabourForceDatasetMetadata,
    FolderMetadataBase,
)
from app.db.session import get_db
from app.services.llm_service import get_embedding_service
from app.services.rag_models import MetadataResult, TableSchema, FolderRetrievalResult

logger = logging.getLogger(__name__)


class RAGService:
    """Hybrid retrieval engine using folder-based metadata with pgvector + full-text search with RRF fusion."""

    # List of (category_name, model_class, table_name) tuples
    FOLDER_TABLES: List[Tuple[str, type, str]] = [
        ("employment", EmploymentDatasetMetadata, "employment_dataset_metadata"),
        ("hours_worked", HoursWorkedDatasetMetadata, "hours_worked_dataset_metadata"),
        ("income", IncomeDatasetMetadata, "income_dataset_metadata"),
        ("labour_force", LabourForceDatasetMetadata, "labour_force_dataset_metadata"),
    ]

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
    ) -> FolderRetrievalResult:
        """Perform hybrid retrieval across folder metadata tables.

        Args:
            query: The search query.
            top_k: Number of results to return.
            category_filter: Optional category to filter by (employment, hours_worked, income, labour_force).
            year_filter: Optional year range dict {"start": int, "end": int}.

        Returns:
            FolderRetrievalResult with metadata and table schemas.
        """
        top_k = top_k or self.hybrid_top_k

        embedding_service = get_embedding_service()
        query_embedding = await embedding_service.embed_query(query)

        async with get_db() as session:
            all_metadata_results: List[MetadataResult] = []

            # Search each folder table (or just the filtered category)
            folders_to_search = self.FOLDER_TABLES
            if category_filter:
                folders_to_search = [
                    (cat, model, table) for cat, model, table in self.FOLDER_TABLES
                    if cat == category_filter
                ]

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
        query_embedding: List[float],
        year_filter: Optional[Dict[str, int]],
    ) -> List[MetadataResult]:
        """Perform hybrid search (vector + full-text + RRF) on a single folder metadata table.

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

        # Run full-text search
        fulltext_results = await self._folder_fulltext_search(
            session, table_name, category, query, year_filter
        )

        # RRF fusion
        fused = self._rrf_fuse_metadata(vector_results, fulltext_results, top_k=10)

        return fused

    async def _folder_vector_search(
        self,
        session: AsyncSession,
        table_name: str,
        category: str,
        query_embedding: List[float],
        year_filter: Optional[Dict[str, int]],
    ) -> List[MetadataResult]:
        """Perform cosine similarity search on folder metadata embeddings."""
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        where_clauses = ["m.embedding IS NOT NULL"]
        params: Dict[str, Any] = {"limit": self.vector_top_k}

        if year_filter:
            if year_filter.get("start"):
                where_clauses.append("(m.year_range->>'max')::int >= :year_start")
                params["year_start"] = year_filter["start"]
            if year_filter.get("end"):
                where_clauses.append("(m.year_range->>'min')::int <= :year_end")
                params["year_end"] = year_filter["end"]

        where_sql = " AND ".join(where_clauses)

        sql = text(f"""
            SELECT m.id, m.file_name, m.file_path, m.table_name, m.description,
                   m.columns, m.primary_dimensions, m.numeric_columns, m.categorical_columns,
                   m.row_count, m.year_range, m.summary_text,
                   m.embedding <=> '{embedding_str}'::vector AS distance
            FROM {table_name} m
            WHERE {where_sql}
            ORDER BY distance
            LIMIT :limit
        """)

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
        year_filter: Optional[Dict[str, int]],
    ) -> List[MetadataResult]:
        """Perform full-text search using tsvector/tsquery on folder metadata."""
        where_clauses = ["m.tsv IS NOT NULL"]
        params: Dict[str, Any] = {"query": query, "limit": self.fulltext_top_k}

        if year_filter:
            if year_filter.get("start"):
                where_clauses.append("(m.year_range->>'max')::int >= :year_start")
                params["year_start"] = year_filter["start"]
            if year_filter.get("end"):
                where_clauses.append("(m.year_range->>'min')::int <= :year_end")
                params["year_end"] = year_filter["end"]

        where_sql = " AND ".join(where_clauses)

        sql = text(f"""
            SELECT m.id, m.file_name, m.file_path, m.table_name, m.description,
                   m.columns, m.primary_dimensions, m.numeric_columns, m.categorical_columns,
                   m.row_count, m.year_range, m.summary_text,
                   ts_rank_cd(m.tsv, plainto_tsquery('english', :query)) AS rank
            FROM {table_name} m
            WHERE {where_sql}
              AND m.tsv @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """)

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

    def _rrf_fuse_metadata(
        self,
        vector_results: List[MetadataResult],
        fulltext_results: List[MetadataResult],
        top_k: int,
    ) -> List[MetadataResult]:
        """Reciprocal Rank Fusion to combine vector and full-text metadata results."""
        scores: Dict[int, float] = defaultdict(float)
        metadata_by_id: Dict[int, MetadataResult] = {}

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
        metadata_results: List[MetadataResult],
    ) -> List[TableSchema]:
        """Fetch full table schemas for SQL generation.

        Args:
            session: Database session
            metadata_results: List of metadata results

        Returns:
            List of TableSchema objects with formatted SQL schema prompts
        """
        table_schemas = []

        # Group by category to minimize queries
        by_category: Dict[str, List[MetadataResult]] = defaultdict(list)
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
                    )
                )

        return table_schemas
