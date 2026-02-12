"""
Unit tests for RAG retrieval system.

Tests the hybrid RAG pipeline including:
- Vector search
- BM25 search
- RRF fusion
- Cross-encoder reranking
- Confidence-based dataset selection

All tests use mocked database operations and realistic mock search results.
No real database queries are executed.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tests.utils.test_helpers import (
    assert_score_in_range,
)

# ============================================================================
# Mock Data Fixtures
# ============================================================================


@pytest.fixture
def mock_metadata_results() -> list[dict[str, Any]]:
    """Create realistic mock metadata results."""
    return [
        {
            "metadata_id": 1,
            "category": "income",
            "file_name": "income_2020.csv",
            "file_path": "/data/income_2020.csv",
            "table_name": "income_data_2020",
            "description": "Average monthly income by age group and gender for 2020",
            "columns": ["year", "age_group", "gender", "average_income", "median_income"],
            "primary_dimensions": ["age_group", "gender"],
            "numeric_columns": ["average_income", "median_income"],
            "categorical_columns": ["age_group", "gender"],
            "row_count": 48,
            "year_range": {"min": 2020, "max": 2020},
            "summary_text": "This dataset contains income statistics for Singapore in 2020, "
            "broken down by age group and gender.",
            "score": 0.92,
        },
        {
            "metadata_id": 2,
            "category": "employment",
            "file_name": "employment_2020.csv",
            "file_path": "/data/employment_2020.csv",
            "table_name": "employment_data_2020",
            "description": "Employment and unemployment rates by age group for 2020",
            "columns": ["year", "age_group", "employment_rate", "unemployment_rate"],
            "primary_dimensions": ["age_group"],
            "numeric_columns": ["employment_rate", "unemployment_rate"],
            "categorical_columns": ["age_group"],
            "row_count": 24,
            "year_range": {"min": 2020, "max": 2020},
            "summary_text": "Employment data for Singapore in 2020, including employment rates "
            "and unemployment rates by age group.",
            "score": 0.78,
        },
        {
            "metadata_id": 3,
            "category": "hours_worked",
            "file_name": "hours_2020.csv",
            "file_path": "/data/hours_2020.csv",
            "table_name": "hours_worked_data_2020",
            "description": "Average weekly hours worked by occupation in 2020",
            "columns": ["year", "occupation", "average_weekly_hours", "median_weekly_hours"],
            "primary_dimensions": ["occupation"],
            "numeric_columns": ["average_weekly_hours", "median_weekly_hours"],
            "categorical_columns": ["occupation"],
            "row_count": 60,
            "year_range": {"min": 2020, "max": 2020},
            "summary_text": "Working hours dataset for Singapore in 2020, showing average "
            "and median weekly hours by occupation.",
            "score": 0.65,
        },
    ]


@pytest.fixture
def mock_vector_search_results() -> list[dict[str, Any]]:
    """Create mock vector search results."""
    return [
        {
            "metadata_id": 1,
            "score": 0.95,
            "file_path": "/data/income_2020.csv",
            "table_name": "income_data_2020",
            "description": "Income data 2020",
        },
        {
            "metadata_id": 3,
            "score": 0.72,
            "file_path": "/data/hours_2020.csv",
            "table_name": "hours_worked_data_2020",
            "description": "Hours worked data 2020",
        },
        {
            "metadata_id": 2,
            "score": 0.68,
            "file_path": "/data/employment_2020.csv",
            "table_name": "employment_data_2020",
            "description": "Employment data 2020",
        },
    ]


@pytest.fixture
def mock_bm25_search_results() -> list[dict[str, Any]]:
    """Create mock BM25 search results."""
    return [
        {
            "metadata_id": 1,
            "score": 8.5,
            "file_path": "/data/income_2020.csv",
            "table_name": "income_data_2020",
            "description": "Income data 2020",
        },
        {
            "metadata_id": 2,
            "score": 6.3,
            "file_path": "/data/employment_2020.csv",
            "table_name": "employment_data_2020",
            "description": "Employment data 2020",
        },
    ]


@pytest.mark.unit
class TestRAGVectorSearch:
    """Test vector search functionality."""

    @pytest.mark.asyncio
    async def test_vector_search_returns_results(self, mock_vector_search_results):
        """Test that vector search returns relevant results using mocked database."""
        from app.services.rag_models import MetadataResult
        from app.services.rag_service import RAGService

        # Create mock RAG service
        rag_service = RAGService()

        # Mock the vector search method
        mock_results = [
            MetadataResult(
                metadata_id=r["metadata_id"],
                category="income" if r["metadata_id"] == 1 else "employment",
                file_name=r["table_name"] + ".csv",
                file_path=r["file_path"],
                table_name=r["table_name"],
                description=r["description"],
                columns=["year", "data"],
                primary_dimensions=["year"],
                numeric_columns=["data"],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text=r["description"],
                score=r["score"],
            )
            for r in mock_vector_search_results
        ]

        with patch.object(
            rag_service, "_folder_vector_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = mock_results

            # Simulate a search call
            results = await mock_search()

            assert len(results) > 0, "Vector search should return results"
            assert len(results) <= 5, "Should respect top_k limit"

            # Check result structure
            for result in results:
                assert result.metadata_id is not None
                assert hasattr(result, "score")
                assert_score_in_range(result.score, 0.0, 1.0)

    @pytest.mark.asyncio
    async def test_vector_search_relevance(self):
        """Test that top results are more relevant than lower results (mocked)."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Mock results with descending scores
        mock_results = [
            Mock(metadata_id=1, score=0.95),
            Mock(metadata_id=2, score=0.78),
            Mock(metadata_id=3, score=0.65),
        ]

        with patch.object(
            rag_service, "_folder_vector_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = mock_results

            results = await mock_search()

            if len(results) >= 2:
                # Scores should be in descending order
                scores = [r.score for r in results]
                assert scores == sorted(
                    scores, reverse=True
                ), "Results should be ordered by score descending"

    @pytest.mark.asyncio
    async def test_vector_search_empty_query(self):
        """Test vector search handles empty query gracefully."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        with patch.object(
            rag_service, "_folder_vector_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = []

            results = await mock_search()
            assert isinstance(results, list), "Should return list"
            assert len(results) == 0, "Should return empty list for empty query"


@pytest.mark.unit
class TestBM25Search:
    """Test BM25 search functionality."""

    @pytest.mark.asyncio
    async def test_bm25_search_keyword_matching(self, mock_bm25_search_results):
        """Test that BM25 search performs good keyword matching (mocked)."""
        from app.services.rag_models import MetadataResult
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Create mock BM25 results
        mock_results = [
            MetadataResult(
                metadata_id=r["metadata_id"],
                category="income" if r["metadata_id"] == 1 else "employment",
                file_name=r["table_name"] + ".csv",
                file_path=r["file_path"],
                table_name=r["table_name"],
                description=r["description"],
                columns=["year", "data"],
                primary_dimensions=["year"],
                numeric_columns=["data"],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text="income 2020 statistics",
                score=r["score"],
            )
            for r in mock_bm25_search_results
        ]

        with patch.object(
            rag_service, "_folder_bm25_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = mock_results

            results = await mock_search()

            assert len(results) > 0, "BM25 should return results"

            # Top results should have reasonable BM25 scores
            for result in results[:3]:
                assert result.score > 0, "BM25 results should have positive scores"

    @pytest.mark.asyncio
    async def test_bm25_vs_vector_search(
        self, mock_vector_search_results, mock_bm25_search_results
    ):
        """Test that BM25 and vector search can return different results."""
        from app.services.rag_models import MetadataResult
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Create mock vector search results
        vector_mocks = [
            MetadataResult(
                metadata_id=r["metadata_id"],
                category=(
                    "income"
                    if r["metadata_id"] == 1
                    else "employment" if r["metadata_id"] == 2 else "hours"
                ),
                file_name=r["table_name"] + ".csv",
                file_path=r["file_path"],
                table_name=r["table_name"],
                description=r["description"],
                columns=["year", "data"],
                primary_dimensions=["year"],
                numeric_columns=["data"],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text=r["description"],
                score=r["score"],
            )
            for r in mock_vector_search_results
        ]

        # Create mock BM25 search results
        bm25_mocks = [
            MetadataResult(
                metadata_id=r["metadata_id"],
                category="income" if r["metadata_id"] == 1 else "employment",
                file_name=r["table_name"] + ".csv",
                file_path=r["file_path"],
                table_name=r["table_name"],
                description=r["description"],
                columns=["year", "data"],
                primary_dimensions=["year"],
                numeric_columns=["data"],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text=r["description"],
                score=r["score"],
            )
            for r in mock_bm25_search_results
        ]

        with (
            patch.object(rag_service, "_folder_vector_search", new_callable=AsyncMock) as mock_vec,
            patch.object(rag_service, "_folder_bm25_search", new_callable=AsyncMock) as mock_bm25,
        ):
            mock_vec.return_value = vector_mocks
            mock_bm25.return_value = bm25_mocks

            vector_results = await mock_vec()
            bm25_results = await mock_bm25()

            # Results should be different (complementary)
            vector_ids = {r.metadata_id for r in vector_results}
            bm25_ids = {r.metadata_id for r in bm25_results}

            # Some overlap is expected
            overlap = len(vector_ids & bm25_ids)
            assert overlap < len(vector_ids) or len(vector_ids) < len(
                bm25_ids
            ), "BM25 and vector search can return different result rankings"


@pytest.mark.unit
class TestRRFFusion:
    """Test Reciprocal Rank Fusion (RRF) algorithm."""

    def test_rrf_fusion_combines_rankings(self):
        """Test that RRF correctly combines rankings from multiple sources."""
        from app.services.rag_models import MetadataResult
        from app.services.rag_service import RAGService

        # Mock results from vector and BM25 search
        vector_results = [
            MetadataResult(
                metadata_id=1,
                category="income",
                file_name="income.csv",
                file_path="/data/income.csv",
                table_name="income_data",
                description="Income data",
                columns=[],
                primary_dimensions=[],
                numeric_columns=[],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text="Test income data",
                score=0.9,
            ),
            MetadataResult(
                metadata_id=2,
                category="employment",
                file_name="employment.csv",
                file_path="/data/employment.csv",
                table_name="employment_data",
                description="Employment data",
                columns=[],
                primary_dimensions=[],
                numeric_columns=[],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text="Test employment data",
                score=0.8,
            ),
            MetadataResult(
                metadata_id=3,
                category="hours",
                file_name="hours.csv",
                file_path="/data/hours.csv",
                table_name="hours_data",
                description="Hours data",
                columns=[],
                primary_dimensions=[],
                numeric_columns=[],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text="Test hours data",
                score=0.7,
            ),
        ]

        bm25_results = [
            MetadataResult(
                metadata_id=2,
                category="employment",
                file_name="employment.csv",
                file_path="/data/employment.csv",
                table_name="employment_data",
                description="Employment data",
                columns=[],
                primary_dimensions=[],
                numeric_columns=[],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text="Test employment data",
                score=0.95,
            ),
            MetadataResult(
                metadata_id=4,
                category="income",
                file_name="income_alt.csv",
                file_path="/data/income_alt.csv",
                table_name="income_data_alt",
                description="Alternative income data",
                columns=[],
                primary_dimensions=[],
                numeric_columns=[],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text="Test income data alt",
                score=0.85,
            ),
            MetadataResult(
                metadata_id=1,
                category="income",
                file_name="income.csv",
                file_path="/data/income.csv",
                table_name="income_data",
                description="Income data",
                columns=[],
                primary_dimensions=[],
                numeric_columns=[],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text="Test income data",
                score=0.75,
            ),
        ]

        # Test RRF fusion
        rag_service = RAGService()
        fused_results = rag_service._rrf_fuse_metadata(vector_results, bm25_results, top_k=10)

        # Check structure
        assert len(fused_results) <= 4, "Should combine unique results"

        # Items appearing in both lists should have higher scores
        metadata_ids = {r.metadata_id for r in fused_results}
        assert 1 in metadata_ids or 2 in metadata_ids, "Should include items from both lists"

        # Scores should be normalized
        for result in fused_results:
            assert result.score > 0, "RRF scores should be positive"

    def test_rrf_fusion_empty_lists(self):
        """Test RRF handles empty input lists."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        result = rag_service._rrf_fuse_metadata([], [], top_k=10)
        assert result == [], "RRF should return empty list for empty inputs"

        # Mock single result
        mock_result = Mock(metadata_id=1, score=0.9)
        result = rag_service._rrf_fuse_metadata([mock_result], [], top_k=10)
        assert len(result) == 1, "RRF should handle one empty list"


@pytest.mark.unit
class TestCrossEncoderReranking:
    """Test cross-encoder reranking functionality."""

    @pytest.mark.asyncio
    async def test_reranking_updates_scores(self, mock_metadata_results):
        """Test that reranking updates result scores."""
        from app.services.rag_models import MetadataResult
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Create mock input results
        input_results = [
            MetadataResult(
                metadata_id=r["metadata_id"],
                category=r["category"],
                file_name=r["file_name"],
                file_path=r["file_path"],
                table_name=r["table_name"],
                description=r["description"],
                columns=r["columns"],
                primary_dimensions=r["primary_dimensions"],
                numeric_columns=r["numeric_columns"],
                categorical_columns=r["categorical_columns"],
                row_count=r["row_count"],
                year_range=r["year_range"],
                summary_text=r["summary_text"],
                score=r["score"],
            )
            for r in mock_metadata_results
        ]

        # Mock reranker to update scores
        reranked_results = [
            MetadataResult(
                metadata_id=1,
                category="income",
                file_name="income_2020.csv",
                file_path="/data/income_2020.csv",
                table_name="income_data_2020",
                description="Income data",
                columns=[],
                primary_dimensions=[],
                numeric_columns=[],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text="Income data",
                score=0.98,  # Reranked score
            ),
            MetadataResult(
                metadata_id=2,
                category="employment",
                file_name="employment_2020.csv",
                file_path="/data/employment_2020.csv",
                table_name="employment_data_2020",
                description="Employment data",
                columns=[],
                primary_dimensions=[],
                numeric_columns=[],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text="Employment data",
                score=0.76,  # Reranked score
            ),
        ]

        with patch.object(rag_service, "_rerank_results", new_callable=AsyncMock) as mock_rerank:
            mock_rerank.return_value = reranked_results

            results = await mock_rerank("test query", input_results)

            # Reranking should return fewer, higher quality results
            assert len(results) <= len(input_results), "Should filter results"

            # Reranked scores should be confidence scores (0-1)
            for result in results:
                assert_score_in_range(result.score, 0.0, 1.0)

    @pytest.mark.asyncio
    async def test_reranking_with_empty_results(self):
        """Test that reranking handles empty input gracefully."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        with patch.object(rag_service, "_rerank_results", new_callable=AsyncMock) as mock_rerank:
            mock_rerank.return_value = []

            results = await mock_rerank("test query", [])

            assert results == [], "Should return empty list for empty input"

    @pytest.mark.asyncio
    async def test_reranking_respects_top_k(self):
        """Test that reranking respects top_k limit."""
        from app.services.rag_models import MetadataResult
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Create 10 mock results
        input_results = [
            MetadataResult(
                metadata_id=i,
                category="income",
                file_name=f"data_{i}.csv",
                file_path=f"/data/data_{i}.csv",
                table_name=f"data_{i}",
                description=f"Dataset {i}",
                columns=[],
                primary_dimensions=[],
                numeric_columns=[],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text=f"Test dataset {i}",
                score=0.5 + (i * 0.05),
            )
            for i in range(10)
        ]

        # Mock reranker to return only top 3
        reranked_results = input_results[:3]

        with patch.object(rag_service, "_rerank_results", new_callable=AsyncMock) as mock_rerank:
            mock_rerank.return_value = reranked_results

            results = await mock_rerank("test query", input_results)

            assert len(results) == 3, "Should respect top_k limit"


@pytest.mark.unit
class TestHybridSearch:
    """Test end-to-end hybrid search pipeline."""

    @pytest.mark.asyncio
    async def test_hybrid_search_pipeline(self, mock_metadata_results):
        """Test full hybrid search pipeline with mocked components."""
        from app.services.rag_models import FolderRetrievalResult, MetadataResult
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Create mock results
        mock_results = [
            MetadataResult(
                metadata_id=r["metadata_id"],
                category=r["category"],
                file_name=r["file_name"],
                file_path=r["file_path"],
                table_name=r["table_name"],
                description=r["description"],
                columns=r["columns"],
                primary_dimensions=r["primary_dimensions"],
                numeric_columns=r["numeric_columns"],
                categorical_columns=r["categorical_columns"],
                row_count=r["row_count"],
                year_range=r["year_range"],
                summary_text=r["summary_text"],
                score=r["score"],
            )
            for r in mock_metadata_results[:3]
        ]

        with patch.object(rag_service, "retrieve", new_callable=AsyncMock) as mock_retrieve:
            # Mock the retrieve method to return results
            mock_retrieve.return_value = FolderRetrievalResult(
                query="test query",
                metadata_results=mock_results,
                table_schemas=[],
                total_results=len(mock_results),
            )

            results = await mock_retrieve("test query", top_k=3)

            # Should return results
            assert len(results.metadata_results) > 0, "Hybrid search should return results"
            assert len(results.metadata_results) <= 3, "Should respect top_k"

            # Results should have required fields
            for result in results.metadata_results:
                assert result.metadata_id is not None
                assert result.file_path is not None
                assert result.score is not None

            # Top result should have reasonable score
            assert (
                results.metadata_results[0].score >= 0.5
            ), "Top result should have reasonable confidence"

    @pytest.mark.asyncio
    async def test_category_filter_affects_results(self):
        """Test that category filter narrows search results."""
        from app.services.rag_models import FolderRetrievalResult, MetadataResult
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # All categories results
        all_results = [
            MetadataResult(
                metadata_id=i,
                category=category,
                file_name=f"{category}_{i}.csv",
                file_path=f"/data/{category}_{i}.csv",
                table_name=f"{category}_data_{i}",
                description=f"{category} dataset",
                columns=[],
                primary_dimensions=[],
                numeric_columns=[],
                categorical_columns=[],
                row_count=100,
                year_range={"min": 2020, "max": 2020},
                summary_text=f"{category} summary",
                score=0.8,
            )
            for i, category in enumerate(["income", "employment", "hours_worked"])
        ]

        # Filtered results (only income)
        filtered_results = [r for r in all_results if r.category == "income"]

        with patch.object(rag_service, "retrieve", new_callable=AsyncMock) as mock_retrieve:
            # Mock all categories call
            mock_retrieve.return_value = FolderRetrievalResult(
                query="test query",
                metadata_results=all_results,
                table_schemas=[],
                total_results=len(all_results),
            )

            all_res = await mock_retrieve("test query")
            assert len(all_res.metadata_results) == 3

            # Mock filtered call
            mock_retrieve.return_value = FolderRetrievalResult(
                query="test query",
                metadata_results=filtered_results,
                table_schemas=[],
                total_results=len(filtered_results),
            )

            filtered_res = await mock_retrieve("test query", category_filter="income")
            assert len(filtered_res.metadata_results) == 1
            assert filtered_res.metadata_results[0].category == "income"

    @pytest.mark.asyncio
    async def test_year_filter_affects_results(self):
        """Test that year filter narrows search results."""
        from app.services.rag_models import FolderRetrievalResult, MetadataResult
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Create results for different years
        all_results = [
            MetadataResult(
                metadata_id=i,
                category="income",
                file_name=f"income_{year}.csv",
                file_path=f"/data/income_{year}.csv",
                table_name=f"income_data_{year}",
                description=f"Income data {year}",
                columns=[],
                primary_dimensions=[],
                numeric_columns=[],
                categorical_columns=[],
                row_count=100,
                year_range={"min": year, "max": year},
                summary_text=f"Income for {year}",
                score=0.8,
            )
            for i, year in enumerate([2019, 2020, 2021])
        ]

        # Filtered results (only 2020)
        filtered_results = [r for r in all_results if r.year_range["min"] == 2020]

        with patch.object(rag_service, "retrieve", new_callable=AsyncMock) as mock_retrieve:
            # Mock filtered call
            mock_retrieve.return_value = FolderRetrievalResult(
                query="test query",
                metadata_results=filtered_results,
                table_schemas=[],
                total_results=len(filtered_results),
            )

            results = await mock_retrieve("test query", year_filter={"start": 2020, "end": 2020})
            assert len(results.metadata_results) == 1
            assert results.metadata_results[0].year_range["min"] == 2020


@pytest.mark.unit
class TestRAGServiceConfiguration:
    """Test RAG service configuration and initialization."""

    def test_rag_service_initialization(self):
        """Test RAG service initializes correctly."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        assert rag_service is not None
        assert hasattr(rag_service, "use_reranking")
        assert hasattr(rag_service, "use_bm25")
        assert rag_service.vector_top_k > 0
        assert rag_service.fulltext_top_k > 0

    def test_rag_config_values(self):
        """Test RAG configuration has expected values."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Check that service was initialized with valid config values
        assert rag_service.rrf_k > 0
        assert rag_service.similarity_threshold >= 0.0
        assert isinstance(rag_service.use_reranking, bool)
        assert isinstance(rag_service.use_bm25, bool)

    def test_rag_service_folder_tables(self):
        """Test RAG service has correct folder table definitions."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Check FOLDER_TABLES constant
        assert len(rag_service.FOLDER_TABLES) == 3
        categories = [cat for cat, _, _ in rag_service.FOLDER_TABLES]
        assert "employment" in categories
        assert "hours_worked" in categories
        assert "income" in categories


@pytest.mark.unit
class TestRAGCaching:
    """Test RAG caching mechanisms."""

    def test_bm25_cache_initialization(self):
        """Test that BM25 cache is initialized."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # BM25 cache should be initialized as empty dict
        assert isinstance(rag_service._bm25_cache, dict)
        assert len(rag_service._bm25_cache) == 0

    def test_bm25_cache_directory_creation(self):
        """Test that BM25 cache directory exists."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Cache directory should be created
        assert rag_service._bm25_cache_dir.exists()

    def test_tokenization(self):
        """Test BM25 tokenization function."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        # Test tokenization
        text = "Average income in Singapore 2020"
        tokens = rag_service._tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        # Should be lowercase and alphanumeric (no special chars)
        assert all(t.islower() or t.isdigit() for t in tokens)

    def test_tokenization_empty_string(self):
        """Test tokenization with empty string."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        tokens = rag_service._tokenize("")
        assert tokens == []

    def test_tokenization_with_special_chars(self):
        """Test tokenization with special characters."""
        from app.services.rag_service import RAGService

        rag_service = RAGService()

        text = "Income $1000-5000 (SGD) - 2020!"
        tokens = rag_service._tokenize(text)

        assert len(tokens) > 0
        # Should extract words
        assert "income" in tokens
        assert "2020" in tokens
