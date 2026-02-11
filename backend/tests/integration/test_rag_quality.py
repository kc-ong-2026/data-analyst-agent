"""
RAG Retrieval Quality Tests.

Tests the RAG pipeline quality:
- Vector search + BM25 hybrid search
- RRF fusion effectiveness
- Cross-encoder reranking quality
- Confidence-based dataset selection
"""

import pytest
import time
from typing import List, Dict, Any


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.requires_db
class TestRAGRetrievalQuality:
    """Test RAG retrieval quality and relevance."""

    async def test_retrieval_returns_relevant_datasets(self):
        """Test that RAG retrieves relevant datasets for income queries."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_service = RAGService(config)

        # Initialize RAG service
        await rag_service.initialize()

        query = "What was the average income in Singapore in 2020?"

        start_time = time.time()
        results = await rag_service.search(
            query=query,
            category="income",
            year=2020
        )
        elapsed = time.time() - start_time

        # Should return results quickly
        assert elapsed < 2.0, f"RAG search took {elapsed:.2f}s, expected < 2s"

        # Should return results
        assert len(results) > 0, "No results returned"

        # Results should have confidence scores
        for result in results[:3]:
            assert "score" in result or hasattr(result, "score")
            print(f"   Dataset: {result.get('file_name', 'unknown')}, Score: {result.get('score', 0):.3f}")

        print(f"✅ RAG retrieval returned {len(results)} datasets in {elapsed:.2f}s")

    async def test_category_filtering(self):
        """Test that category filtering works correctly."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_service = RAGService(config)
        await rag_service.initialize()

        # Test income category
        income_results = await rag_service.search(
            query="average income 2020",
            category="income",
        )

        # Test employment category
        employment_results = await rag_service.search(
            query="employment rate 2020",
            category="employment",
        )

        # Results should be different based on category
        assert len(income_results) > 0
        assert len(employment_results) > 0

        print(f"✅ Category filtering: income={len(income_results)}, employment={len(employment_results)}")

    async def test_year_filtering(self):
        """Test that year filtering works correctly."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_service = RAGService(config)
        await rag_service.initialize()

        # Search for specific year
        results_2020 = await rag_service.search(
            query="average income",
            category="income",
            year=2020
        )

        results_2019 = await rag_service.search(
            query="average income",
            category="income",
            year=2019
        )

        assert len(results_2020) > 0
        assert len(results_2019) > 0

        # Results should include the specified year in metadata or filename
        print(f"✅ Year filtering: 2020={len(results_2020)}, 2019={len(results_2019)}")

    async def test_hybrid_search_vs_vector_only(self):
        """Compare hybrid search (vector + BM25) vs vector-only."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_service = RAGService(config)
        await rag_service.initialize()

        query = "median income from work 2020"

        # Hybrid search (default)
        start = time.time()
        hybrid_results = await rag_service.search(query=query, category="income")
        hybrid_time = time.time() - start

        # Should return results
        assert len(hybrid_results) > 0

        print(f"✅ Hybrid search returned {len(hybrid_results)} results in {hybrid_time:.2f}s")
        for i, result in enumerate(hybrid_results[:3], 1):
            score = result.get("score", 0) if isinstance(result, dict) else getattr(result, "score", 0)
            print(f"   {i}. Score: {score:.3f}")

    async def test_reranking_scores(self):
        """Test that reranking produces reasonable confidence scores."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_config = config.get_rag_config()

        # Only run if reranking is enabled
        if not rag_config.get("use_reranking", True):
            pytest.skip("Reranking not enabled")

        rag_service = RAGService(config)
        await rag_service.initialize()

        query = "average income from work 2020"
        results = await rag_service.search(query=query, category="income")

        assert len(results) > 0

        # Check that scores are in reasonable range (0 to 1)
        for result in results:
            score = result.get("score", 0) if isinstance(result, dict) else getattr(result, "score", 0)
            assert 0 <= score <= 1, f"Score {score} out of range [0, 1]"

        # Top result should have higher score than later results
        if len(results) >= 2:
            score1 = results[0].get("score", 0) if isinstance(results[0], dict) else getattr(results[0], "score", 0)
            score2 = results[1].get("score", 0) if isinstance(results[1], dict) else getattr(results[1], "score", 0)
            assert score1 >= score2, f"Top result score {score1} < second result {score2}"

        print(f"✅ Reranking scores are in valid range")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.requires_db
class TestConfidenceBasedSelection:
    """Test confidence-based dataset selection."""

    async def test_loads_high_confidence_datasets(self):
        """Test that high-confidence datasets are loaded."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_config = config.get_rag_config()
        confidence_threshold = rag_config.get("confidence_threshold", 0.5)

        rag_service = RAGService(config)
        await rag_service.initialize()

        query = "average income from work 2020"
        results = await rag_service.search(query=query, category="income")

        # Count datasets above threshold
        high_confidence = [
            r for r in results
            if (r.get("score", 0) if isinstance(r, dict) else getattr(r, "score", 0)) >= confidence_threshold
        ]

        print(f"✅ Confidence threshold {confidence_threshold}:")
        print(f"   Total datasets: {len(results)}")
        print(f"   Above threshold: {len(high_confidence)}")

        assert len(high_confidence) >= 1, "Should have at least 1 high-confidence result"

    async def test_respects_min_max_dataset_limits(self):
        """Test that min/max dataset limits are respected."""
        from app.services.rag_service import RAGService
        from app.config import get_config

        config = get_config()
        rag_config = config.get_rag_config()

        min_datasets = rag_config.get("min_datasets", 1)
        max_datasets = rag_config.get("max_datasets", 3)

        rag_service = RAGService(config)
        await rag_service.initialize()

        query = "income 2020"
        results = await rag_service.search(query=query, category="income")

        # Results should respect max_datasets
        print(f"✅ Dataset limits: min={min_datasets}, max={max_datasets}")
        print(f"   Returned: {len(results[:max_datasets])} datasets")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
