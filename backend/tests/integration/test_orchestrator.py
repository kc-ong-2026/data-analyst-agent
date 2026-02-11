"""
Orchestrator Integration Tests (Mocked).

Tests the multi-agent workflow through the orchestrator using mocked LLM responses.
Fast and deterministic - no real LLM API calls.
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from tests.utils.test_helpers import PerformanceTimer, assert_dict_contains_keys


@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.asyncio
class TestOrchestratorMocked:
    """Test orchestrator with mocked LLM responses."""

    async def test_orchestrator_handles_simple_query(self, mock_orchestrator):
        """Test orchestrator processes simple query."""
        query = "What was the average income in 2020?"

        with PerformanceTimer() as timer:
            result = await mock_orchestrator.process_query(query)

        print(f"\nProcessed query in {timer.elapsed_seconds:.2f}s")

        assert result is not None
        assert result["success"] is True
        assert "answer" in result

    async def test_orchestrator_handles_visualization_request(self, mock_orchestrator):
        """Test orchestrator with visualization request."""
        query = "Show me a bar chart of employment rates"

        result = await mock_orchestrator.process_query(query)

        assert result is not None
        assert result["success"] is True

        # Check for visualization in response
        if "visualization" in result and result["visualization"]:
            assert "type" in result["visualization"]

    async def test_orchestrator_rejects_off_topic(self, mock_orchestrator):
        """Test orchestrator rejects off-topic queries."""
        query = "What is the weather like today?"

        result = await mock_orchestrator.process_query(query)

        assert result is not None
        assert result["success"] is False

    async def test_orchestrator_handles_empty_query(self, mock_orchestrator):
        """Test orchestrator handles empty query gracefully."""
        query = ""

        result = await mock_orchestrator.process_query(query)

        assert result is not None
        assert result["success"] is False


@pytest.mark.integration
@pytest.mark.fast
class TestOrchestratorInitialization:
    """Test orchestrator initialization without LLM calls."""

    def test_orchestrator_can_be_imported(self):
        """Test that orchestrator module can be imported."""
        from app.services.agents.orchestrator import AgentOrchestrator
        assert AgentOrchestrator is not None

    def test_get_orchestrator_function_exists(self):
        """Test that get_orchestrator helper exists."""
        from app.services.agents import get_orchestrator
        assert callable(get_orchestrator)


@pytest.mark.integration
@pytest.mark.fast
class TestAgentStateManagement:
    """Test agent state management without LLM calls."""

    def test_agent_state_creation(self):
        """Test AgentState can be created."""
        from app.services.agents import AgentState

        state = AgentState()
        assert state is not None
        assert hasattr(state, 'messages')
        assert hasattr(state, 'metadata')

    def test_agent_state_add_message(self):
        """Test adding messages to AgentState."""
        from app.services.agents import AgentState

        state = AgentState()
        state.add_message("user", "Test message")

        assert len(state.messages) > 0
        assert state.messages[0].content == "Test message"


@pytest.mark.integration
@pytest.mark.fast
class TestAgentInitialization:
    """Test individual agents can be initialized."""

    def test_verification_agent_initializes(self):
        """Test verification agent initialization."""
        from app.services.agents.verification import QueryVerificationAgent
        from app.config import get_config

        config = get_config()
        agent = QueryVerificationAgent(config)

        assert agent is not None
        print("✅ Verification agent initialized")

    def test_coordinator_agent_initializes(self):
        """Test coordinator agent initialization."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()
        agent = DataCoordinatorAgent(config)

        assert agent is not None
        print("✅ Coordinator agent initialized")

    def test_extraction_agent_initializes(self):
        """Test extraction agent initialization."""
        from app.services.agents.extraction import DataExtractionAgent
        from app.config import get_config

        config = get_config()
        agent = DataExtractionAgent(config)

        assert agent is not None
        print("✅ Extraction agent initialized")

    def test_analytics_agent_initializes(self):
        """Test analytics agent initialization."""
        from app.services.agents.analytics import AnalyticsAgent
        from app.config import get_config

        config = get_config()
        agent = AnalyticsAgent(config)

        assert agent is not None
        print("✅ Analytics agent initialized")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
