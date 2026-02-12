"""
End-to-End Pipeline Integration Tests (Mocked).

Tests the complete query → response flow through all agents using mocked LLM responses.
No real LLM API calls - fast and deterministic.
"""

import time

import pytest


@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.asyncio
class TestEndToEndPipeline:
    """Test complete pipeline with mocked LLM responses."""

    async def test_simple_income_query(self, mock_orchestrator):
        """Test simple income query flows through all agents."""
        query = "What was the average income in Singapore in 2020?"

        start_time = time.time()
        result = await mock_orchestrator.execute(query)
        elapsed = time.time() - start_time

        # Verify response structure
        assert result is not None
        assert result["success"] is True
        assert "answer" in result

        # Should be very fast with mocks
        assert elapsed < 1.0, f"Mocked query took {elapsed:.2f}s, expected < 1s"

        print(f"✅ Simple income query completed in {elapsed:.3f}s")

    async def test_employment_query(self, mock_orchestrator):
        """Test employment query with year specification."""
        query = "Show me employment rates in Singapore for 2019 and 2020"

        start_time = time.time()
        result = await mock_orchestrator.execute(query)
        elapsed = time.time() - start_time

        assert result is not None
        assert result["success"] is True
        assert elapsed < 1.0

        print(f"✅ Employment query completed in {elapsed:.3f}s")

    async def test_comparison_query(self, mock_orchestrator):
        """Test query requiring comparison across years."""
        query = "Compare average income between 2019 and 2020"

        start_time = time.time()
        result = await mock_orchestrator.execute(query)
        elapsed = time.time() - start_time

        assert result is not None
        assert result["success"] is True
        assert elapsed < 1.0

        print(f"✅ Comparison query completed in {elapsed:.3f}s")

    async def test_off_topic_rejection(self, mock_orchestrator):
        """Test that off-topic queries are rejected."""
        query = "What's the weather like in Singapore?"

        start_time = time.time()
        result = await mock_orchestrator.execute(query)
        elapsed = time.time() - start_time

        # Should reject off-topic query
        assert result is not None
        assert result["success"] is False
        assert "error" in result or "message" in result

        print(f"✅ Off-topic query rejected in {elapsed:.3f}s")

    async def test_missing_year_handling(self, mock_orchestrator):
        """Test query without year specification."""
        query = "What was the average income in Singapore?"

        start_time = time.time()
        result = await mock_orchestrator.execute(query)
        elapsed = time.time() - start_time

        # Should handle gracefully
        assert result is not None
        assert elapsed < 1.0

        print(f"✅ Missing year query handled in {elapsed:.3f}s")

    async def test_visualization_request(self, mock_orchestrator):
        """Test query requesting visualization."""
        query = "Show me a bar chart of income by age group in 2020"

        result = await mock_orchestrator.execute(query)

        assert result is not None
        assert result["success"] is True

        # Should include visualization
        if "visualization" in result and result["visualization"]:
            viz = result["visualization"]
            assert "type" in viz
            assert viz["type"] in ["bar", "line", "pie"]
            print(f"✅ Visualization request handled: {viz['type']} chart")


@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.asyncio
class TestPipelineErrorHandling:
    """Test error handling and graceful degradation."""

    async def test_handles_invalid_year(self, mock_orchestrator):
        """Test handling of invalid year specification."""
        query = "What was the average income in Singapore in 2050?"

        result = await mock_orchestrator.execute(query)

        # Should handle gracefully, not crash
        assert result is not None
        print("✅ Invalid year handled gracefully")

    async def test_handles_ambiguous_query(self, mock_orchestrator):
        """Test handling of ambiguous queries."""
        query = "Show me the data"

        result = await mock_orchestrator.execute(query)

        # Should handle gracefully
        assert result is not None
        print("✅ Ambiguous query handled gracefully")

    async def test_handles_empty_query(self, mock_orchestrator):
        """Test handling of empty query."""
        query = ""

        result = await mock_orchestrator.execute(query)

        # Should handle gracefully
        assert result is not None
        assert result["success"] is False
        print("✅ Empty query handled gracefully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
