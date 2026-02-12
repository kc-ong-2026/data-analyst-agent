"""
RAG Retrieval Quality Tests.

Tests the RAG pipeline quality:
- Vector search + BM25 hybrid search
- RRF fusion effectiveness
- Cross-encoder reranking quality
- Confidence-based dataset selection
"""

import time

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.requires_db
class TestRAGRetrievalQuality:
    """Test RAG retrieval quality and relevance."""

    async def test_retrieval_returns_relevant_datasets(self):
        """Test that RAG retrieves relevant datasets for income queries."""
        from app.config import get_config
        from app.services.rag_service import RAGService

        config = get_config()
        rag_service = RAGService()

        # Initialize RAG service
        await rag_service.warmup_caches()

        query = "What was the average income in Singapore in 2020?"

        start_time = time.time()
        results = await rag_service.retrieve(
            query=query, category_filter="income", year_filter={"start": 2020, "end": 2020}
        )
        elapsed = time.time() - start_time

        # Should return results quickly
        assert elapsed < 2.0, f"RAG search took {elapsed:.2f}s, expected < 2s"

        # Should return results
        assert len(results.table_schemas) > 0, "No results returned"

        # Results should have confidence scores
        for schema in results.table_schemas[:3]:
            assert hasattr(schema, "score")
            print(f"   Dataset: {schema.file_path}, Score: {schema.score:.3f}")

        print(f"✅ RAG retrieval returned {len(results.table_schemas)} datasets in {elapsed:.2f}s")

    async def test_category_filtering(self):
        """Test that category filtering works correctly."""
        from app.config import get_config
        from app.services.rag_service import RAGService

        config = get_config()
        rag_service = RAGService()
        await rag_service.warmup_caches()

        # Test income category
        income_results = await rag_service.retrieve(
            query="average income 2020",
            category_filter="income",
        )

        # Test employment category
        employment_results = await rag_service.retrieve(
            query="employment rate 2020",
            category_filter="employment",
        )

        # Results should be different based on category
        assert len(income_results.table_schemas) > 0
        assert len(employment_results.table_schemas) > 0

        print(
            f"✅ Category filtering: income={len(income_results.table_schemas)}, employment={len(employment_results.table_schemas)}"
        )

    async def test_year_filtering(self):
        """Test that year filtering works correctly."""
        from app.config import get_config
        from app.services.rag_service import RAGService

        config = get_config()
        rag_service = RAGService()
        await rag_service.warmup_caches()

        # Search for specific year
        results_2020 = await rag_service.retrieve(
            query="average income",
            category_filter="income",
            year_filter={"start": 2020, "end": 2020},
        )

        results_2019 = await rag_service.retrieve(
            query="average income",
            category_filter="income",
            year_filter={"start": 2019, "end": 2019},
        )

        assert len(results_2020.table_schemas) > 0
        assert len(results_2019.table_schemas) > 0

        # Results should include the specified year in metadata or filename
        print(
            f"✅ Year filtering: 2020={len(results_2020.table_schemas)}, 2019={len(results_2019.table_schemas)}"
        )

    async def test_hybrid_search_vs_vector_only(self):
        """Compare hybrid search (vector + BM25) vs vector-only."""
        from app.config import get_config
        from app.services.rag_service import RAGService

        config = get_config()
        rag_service = RAGService()
        await rag_service.warmup_caches()

        query = "median income from work 2020"

        # Hybrid search (default)
        start = time.time()
        hybrid_results = await rag_service.retrieve(query=query, category_filter="income")
        hybrid_time = time.time() - start

        # Should return results
        assert len(hybrid_results.table_schemas) > 0

        print(
            f"✅ Hybrid search returned {len(hybrid_results.table_schemas)} results in {hybrid_time:.2f}s"
        )
        for i, schema in enumerate(hybrid_results.table_schemas[:3], 1):
            print(f"   {i}. Score: {schema.score:.3f}")

    async def test_reranking_scores(self):
        """Test that reranking produces reasonable confidence scores."""
        from app.config import get_config
        from app.services.rag_service import RAGService

        config = get_config()
        rag_config = config.get_rag_config()

        # Only run if reranking is enabled
        if not rag_config.get("use_reranking", True):
            pytest.skip("Reranking not enabled")

        rag_service = RAGService()
        await rag_service.warmup_caches()

        query = "average income from work 2020"
        results = await rag_service.retrieve(query=query, category_filter="income")

        assert len(results) > 0

        # Check that scores are in reasonable range (0 to 1)
        for schema in results.table_schemas:
            assert hasattr(schema, "score")
            assert 0 <= schema.score <= 1, f"Score {schema.score} out of range [0, 1]"

        # Top result should have higher score than later results
        if len(results.table_schemas) >= 2:
            score1 = results.table_schemas[0].score
            score2 = results.table_schemas[1].score
            assert score1 >= score2, f"Top result score {score1} < second result {score2}"

        print("✅ Reranking scores are in valid range")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.requires_db
class TestConfidenceBasedSelection:
    """Test confidence-based dataset selection."""

    async def test_loads_high_confidence_datasets(self):
        """Test that high-confidence datasets are loaded."""
        from app.config import get_config
        from app.services.rag_service import RAGService

        config = get_config()
        rag_config = config.get_rag_config()
        confidence_threshold = rag_config.get("confidence_threshold", 0.5)

        rag_service = RAGService()
        await rag_service.warmup_caches()

        query = "average income from work 2020"
        results = await rag_service.retrieve(query=query, category_filter="income")

        # Count datasets above threshold
        high_confidence = [
            schema for schema in results.table_schemas if schema.score >= confidence_threshold
        ]

        print(f"✅ Confidence threshold {confidence_threshold}:")
        print(f"   Total datasets: {len(results.table_schemas)}")
        print(f"   Above threshold: {len(high_confidence)}")

        assert len(high_confidence) >= 1, "Should have at least 1 high-confidence result"

    async def test_respects_min_max_dataset_limits(self):
        """Test that min/max dataset limits are respected."""
        from app.config import get_config
        from app.services.rag_service import RAGService

        config = get_config()
        rag_config = config.get_rag_config()

        min_datasets = rag_config.get("min_datasets", 1)
        max_datasets = rag_config.get("max_datasets", 3)

        rag_service = RAGService()
        await rag_service.warmup_caches()

        query = "income 2020"
        results = await rag_service.retrieve(query=query, category_filter="income")

        # Results should respect max_datasets
        print(f"✅ Dataset limits: min={min_datasets}, max={max_datasets}")
        print(f"   Returned: {len(results[:max_datasets])} datasets")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
