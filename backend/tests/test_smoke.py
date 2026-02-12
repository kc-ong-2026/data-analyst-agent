"""
Smoke tests to verify Docker test environment is working.

These are minimal tests to validate:
- Import system works
- Database connection works
- Basic agent instantiation works
"""

import pytest


def test_import_config():
    """Test that config module imports correctly."""
    from app.config import get_config

    config = get_config()
    assert config is not None


def test_import_agents():
    """Test that all agents can be imported."""
    from app.services.agents import (
        AgentOrchestrator,
        AnalyticsAgent,
        DataCoordinatorAgent,
        DataExtractionAgent,
        QueryVerificationAgent,
    )

    assert QueryVerificationAgent is not None
    assert DataCoordinatorAgent is not None
    assert DataExtractionAgent is not None
    assert AnalyticsAgent is not None
    assert AgentOrchestrator is not None


def test_import_base_classes():
    """Test that base classes can be imported."""
    from app.services.agents import (
        AgentResponse,
        AgentState,
        BaseAgent,
        GraphState,
    )

    assert BaseAgent is not None
    assert AgentState is not None
    assert AgentResponse is not None
    assert GraphState is not None


@pytest.mark.asyncio
async def test_agent_state_creation():
    """Test that AgentState can be created and converted."""
    from langchain_core.messages import HumanMessage

    from app.services.agents import AgentState

    state = AgentState()
    state.current_task = "test task"
    state.messages.append(HumanMessage(content="Test message"))

    # Convert to GraphState
    graph_state = state.to_graph_state()
    assert graph_state is not None
    assert graph_state["current_task"] == "test task"
    assert len(graph_state["messages"]) == 1

    # Convert back to AgentState
    state2 = AgentState.from_graph_state(graph_state)
    assert state2.current_task == "test task"
    assert len(state2.messages) == 1


@pytest.mark.asyncio
async def test_database_connection(async_db_session):
    """Test that database connection works."""
    # This test uses the async_db_session fixture from conftest.py
    # If it passes, it means the database is accessible
    assert async_db_session is not None


def test_llm_service_import():
    """Test that LLM service can be imported."""
    from app.services.llm_service import LLMService, get_llm_service

    assert LLMService is not None
    assert get_llm_service is not None


def test_rag_service_import():
    """Test that RAG service can be imported."""
    try:
        from app.services.rag_service import RAGService

        assert RAGService is not None
    except ImportError:
        # RAG service might not exist yet
        pytest.skip("RAG service not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
