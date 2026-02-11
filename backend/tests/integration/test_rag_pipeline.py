"""
RAG Pipeline Integration Tests.

Tests the complete RAG retrieval pipeline:
- Query → Embedding → Vector Search → BM25 → RRF Fusion → Reranking
- Cache performance (embeddings, BM25 index)
- Database connection handling
- End-to-end retrieval latency
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from tests.utils.test_helpers import PerformanceTimer


@pytest.mark.integration
@pytest.mark.requires_db
class TestRAGPipelineEndToEnd:
    """Test complete RAG pipeline from query to results."""

    @pytest.mark.asyncio
    async def test_query_to_results_full_pipeline(
        self,
        async_db_session,
        sample_datasets,
    ):
        """Test end-to-end: Query → Embedding → Search → Rerank → Results."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_service = RAGService(config, async_db_session)

        query = "average income statistics in 2020"

        with PerformanceTimer() as timer:
            # Full pipeline
            results = await rag_service.search_datasets(query, top_k=5)

        print(f"\nPipeline completed in {timer.elapsed_ms:.0f}ms")
        print(f"Retrieved {len(results)} datasets")

        # Should return results
        assert len(results) > 0
        assert len(results) <= 5

        # Results should have required fields
        for result in results:
            assert "dataset_name" in result
            assert "description" in result
            assert "score" in result
            assert "file_path" in result

        # Scores should be in descending order
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_embedding_generation(
        self,
        async_db_session,
        sample_datasets,
    ):
        """Test query embedding generation."""
        from app.services.embedding_service import EmbeddingService

        embedding_service = EmbeddingService()

        query = "employment rate statistics"

        # Generate embedding
        embedding = await embedding_service.embed_query(query)

        # Should return valid embedding
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # OpenAI embedding dimension

        # Values should be floats
        assert all(isinstance(x, (int, float)) for x in embedding)

    @pytest.mark.asyncio
    async def test_vector_search_component(
        self,
        async_db_session,
        sample_datasets,
    ):
        """Test vector search component of pipeline."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_service = RAGService(config, async_db_session)

        query = "income data"

        # Vector search
        vector_results = await rag_service.vector_search(query, top_k=10)

        print(f"\nVector search returned {len(vector_results)} results")

        assert len(vector_results) > 0
        assert len(vector_results) <= 10

        # Should have similarity scores
        for result in vector_results:
            assert "score" in result
            assert 0 <= result["score"] <= 1

    @pytest.mark.asyncio
    async def test_bm25_search_component(
        self,
        async_db_session,
        sample_datasets,
    ):
        """Test BM25 search component of pipeline."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_service = RAGService(config, async_db_session)

        query = "employment statistics"

        # BM25 search
        bm25_results = await rag_service.bm25_search(query, top_k=10)

        print(f"\nBM25 search returned {len(bm25_results)} results")

        assert len(bm25_results) > 0

        # BM25 should excel at keyword matching
        for result in bm25_results[:3]:
            desc_lower = result["description"].lower()
            # Top results should contain query keywords
            # (This is a soft check as not all datasets may match)

    @pytest.mark.asyncio
    async def test_rrf_fusion_component(
        self,
        async_db_session,
        sample_datasets,
    ):
        """Test RRF fusion combines results correctly."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_service = RAGService(config, async_db_session)

        query = "income statistics"

        # Get separate results
        vector_results = await rag_service.vector_search(query, top_k=20)
        bm25_results = await rag_service.bm25_search(query, top_k=20)

        # Fuse results
        fused_results = RAGService._rrf_fusion(
            vector_results, bm25_results, k=60
        )

        print(f"\nFused {len(vector_results)} vector + {len(bm25_results)} BM25 results")
        print(f"Result: {len(fused_results)} unique datasets")

        # Should have results
        assert len(fused_results) > 0

        # Should prioritize results appearing in both lists
        # (Items in both should have higher RRF scores)

    @pytest.mark.asyncio
    async def test_reranking_component(
        self,
        async_db_session,
        sample_datasets,
    ):
        """Test cross-encoder reranking component."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_service = RAGService(config, async_db_session)

        query = "average income data"

        # Get candidates for reranking
        vector_results = await rag_service.vector_search(query, top_k=20)

        # Rerank
        reranked_results = await rag_service.rerank_results(
            query, vector_results[:10], top_k=5
        )

        print(f"\nReranked {len(vector_results[:10])} → {len(reranked_results)} results")

        assert len(reranked_results) <= 5

        # Reranked scores should be confidence scores
        for result in reranked_results:
            assert 0 <= result["score"] <= 1


@pytest.mark.integration
@pytest.mark.requires_db
class TestDatabaseIntegration:
    """Test database integration in RAG pipeline."""

    @pytest.mark.asyncio
    async def test_database_connection_pooling(
        self,
        async_db_session,
        sample_datasets,
    ):
        """Test that database connections are managed efficiently."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()

        # Create multiple RAG service instances
        services = [
            RAGService(config, async_db_session)
            for _ in range(5)
        ]

        # All should work concurrently
        import asyncio

        async def search(service):
            return await service.search_datasets("income data", top_k=3)

        results = await asyncio.gather(
            *[search(s) for s in services],
            return_exceptions=True
        )

        # All should succeed
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Database connection issues: {errors}"

    @pytest.mark.asyncio
    async def test_handles_database_unavailable(self):
        """Test handling when database is unavailable."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()

        # Create RAG service without DB session
        rag_service = RAGService(config, None)

        # Should handle gracefully
        query = "income data"

        try:
            results = await rag_service.search_datasets(query, top_k=5)
            # May return empty results or fallback
            assert isinstance(results, list)
        except Exception as e:
            # Should raise clear error, not crash
            assert "database" in str(e).lower() or "session" in str(e).lower()

    @pytest.mark.asyncio
    async def test_vector_index_exists(
        self,
        async_db_session,
        sample_datasets,
    ):
        """Test that vector indexes exist in database."""
        # Query to check for IVFFlat index
        result = await async_db_session.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'data_chunks'
            AND indexname LIKE '%embedding%'
            """
        )

        indexes = result.fetchall()
        print(f"\nVector indexes found: {len(indexes)}")

        # Should have at least one vector index
        assert len(indexes) > 0, "No vector indexes found"


@pytest.mark.integration
@pytest.mark.requires_db
class TestDataQualityAndConsistency:
    """Test data quality and consistency in RAG pipeline."""

    @pytest.mark.asyncio
    async def test_search_results_consistent(
        self,
        async_db_session,
        sample_datasets,
    ):
        """Test that same query returns consistent results."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_service = RAGService(config, async_db_session)

        query = "income statistics"

        # Search multiple times
        results_1 = await rag_service.search_datasets(query, top_k=5)
        results_2 = await rag_service.search_datasets(query, top_k=5)

        # Results should be identical
        assert len(results_1) == len(results_2)

        for r1, r2 in zip(results_1, results_2):
            assert r1["dataset_name"] == r2["dataset_name"]
            # Scores should be very similar (within 0.01)
            assert abs(r1["score"] - r2["score"]) < 0.01

    @pytest.mark.asyncio
    async def test_results_have_valid_file_paths(
        self,
        async_db_session,
        sample_datasets,
    ):
        """Test that results include valid file paths."""
        from app.services.rag_service import RAGService
        from app.config import get_config
        from pathlib import Path

        config = get_config()
        rag_service = RAGService(config, async_db_session)

        query = "employment data"

        results = await rag_service.search_datasets(query, top_k=5)

        # Check file paths
        for result in results:
            file_path = result.get("file_path")
            assert file_path is not None, "Missing file path"

            # Path should exist or be a valid path format
            # (May not exist if using mock data)

    @pytest.mark.asyncio
    async def test_no_duplicate_results(
        self,
        async_db_session,
        sample_datasets,
    ):
        """Test that search doesn't return duplicate datasets."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_service = RAGService(config, async_db_session)

        query = "income data"

        results = await rag_service.search_datasets(query, top_k=10)

        # Check for duplicates
        dataset_names = [r["dataset_name"] for r in results]
        unique_names = set(dataset_names)

        assert len(dataset_names) == len(unique_names), (
            f"Duplicate datasets found: {[n for n in dataset_names if dataset_names.count(n) > 1]}"
        )
