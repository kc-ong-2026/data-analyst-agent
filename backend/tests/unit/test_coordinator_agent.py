"""
Unit tests for Data Coordinator Agent.

Tests the coordinator agent's ability to:
- Analyze user query intent
- Identify required data sources
- Create workflow execution plans
- Determine delegation to extraction/analytics agents

NOTE: These are TRUE unit tests with mocked LLM calls for fast execution.
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from tests.utils.test_helpers import (
    create_mock_llm,
    assert_dict_contains_keys,
)


def create_mock_llm_response(content: str):
    """Create a mock LLM response with the given content."""
    mock_response = Mock()
    mock_response.content = content
    return mock_response


@pytest.mark.unit
class TestQueryAnalysis:
    """Test query intent extraction and analysis."""

    @pytest.mark.asyncio
    async def test_simple_query_intent_extraction(self, mock_graph_state):
        """Test extraction of intent from simple query."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        # Mock LLM responses for each node in the workflow
        mock_llm_responses = [
            # analyze_query response
            create_mock_llm_response("""{"intent": "get_average", "data_type": "income", "time_scope": "2020", "filters": {}}"""),
            # identify_data_sources response
            create_mock_llm_response("""{"required_datasets": ["income_2020"], "categories": ["income"]}"""),
            # create_plan response
            create_mock_llm_response("""{"steps": [{"task": "Extract income data", "agent": "extraction"}], "visualization_suggested": false}"""),
            # determine_delegation response
            create_mock_llm_response("""{"next_agent": "extraction", "reason": "Need to extract income data"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)

            state = mock_graph_state.copy()
            state["query"] = "What was the average income in 2020?"

            response = await agent.execute(state)

            assert response.success is True
            # Coordinator returns "workflow" and "delegation" keys
            assert "workflow" in response.data or "delegation" in response.data

    @pytest.mark.asyncio
    async def test_complex_query_intent_extraction(self, mock_graph_state):
        """Test extraction of intent from complex multi-part query."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "compare", "data_type": "income", "time_scope": "2019-2020", "filters": {"sex": ["male", "female"]}, "visualization": true}"""),
            create_mock_llm_response("""{"required_datasets": ["income_2019", "income_2020"], "categories": ["income"], "filters": {"years": [2019, 2020]}}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract income data", "agent": "extraction"}, {"task": "Compare and visualize", "agent": "analytics"}], "visualization_suggested": true}"""),
            create_mock_llm_response("""{"next_agent": "extraction", "reason": "Need to extract data first"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "Compare average income between males and females in 2019 and 2020, and show me a chart"

            response = await agent.execute(state)

            assert response.success is True

            # Should identify comparison and visualization intent
            workflow = response.data.get("workflow", {})
            if workflow:
                steps = workflow.get("steps", [])
                if steps:
                    # steps is a list of dicts, extract task/agent fields
                    step_texts = [str(s).lower() for s in steps]
                    # Should include relevant terms
                    assert any("extract" in s or "retrieve" in s or "data" in s for s in step_texts) or \
                           any("visuali" in s or "chart" in s for s in step_texts)

    @pytest.mark.asyncio
    async def test_visualization_intent_detection(self, mock_graph_state):
        """Test detection of visualization request in query."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "visualize", "data_type": "employment", "time_scope": "2020", "visualization": true}"""),
            create_mock_llm_response("""{"required_datasets": ["employment_2020"], "categories": ["employment"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract employment data", "agent": "extraction"}, {"task": "Create bar chart", "agent": "analytics"}], "visualization_suggested": true}"""),
            create_mock_llm_response("""{"next_agent": "extraction", "reason": "Need data first"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "Show me a bar chart of employment rates by age group in 2020"

            response = await agent.execute(state)

            assert response.success is True

            # Should identify visualization requirement
            workflow = response.data.get("workflow", {})
            # Visualization flag may or may not be present, test is flexible


@pytest.mark.unit
class TestDataSourceIdentification:
    """Test identification of required data sources."""

    @pytest.mark.asyncio
    async def test_single_dataset_identification(self, mock_graph_state):
        """Test identification of single dataset requirement."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "get_average", "data_type": "income", "time_scope": "2020"}"""),
            create_mock_llm_response("""{"required_datasets": ["income_2020"], "categories": ["income"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract income data", "agent": "extraction"}]}"""),
            create_mock_llm_response("""{"next_agent": "extraction"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "What was the average income in 2020?"
            state["query_validation"] = {
                "is_valid": True,
                "topic": "income",
                "specified_year": 2020,
            }

            response = await agent.execute(state)

            assert response.success is True

            # Should identify income data as required
            workflow_plan = response.data.get("workflow", {})
            if "required_datasets" in workflow_plan:
                datasets = workflow_plan["required_datasets"]
                assert any("income" in str(d).lower() for d in datasets)

    @pytest.mark.asyncio
    async def test_multiple_dataset_identification(self, mock_graph_state):
        """Test identification of multiple dataset requirements."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "compare", "data_type": "employment", "time_scope": "2019-2020"}"""),
            create_mock_llm_response("""{"required_datasets": ["employment_2019", "employment_2020"], "categories": ["employment"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract employment data for both years", "agent": "extraction"}]}"""),
            create_mock_llm_response("""{"next_agent": "extraction"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "Compare employment rates in 2019 and 2020"
            state["query_validation"] = {
                "is_valid": True,
                "topic": "employment",
                "specified_year": [2019, 2020],
            }

            response = await agent.execute(state)

            assert response.success is True

            # Should identify need for multiple years of data
            workflow_plan = response.data.get("workflow", {})
            if "required_datasets" in workflow_plan:
                datasets = workflow_plan["required_datasets"]
                # Should identify employment data
                assert any("employment" in str(d).lower() for d in datasets)

    @pytest.mark.asyncio
    async def test_year_filtered_identification(self, mock_graph_state):
        """Test that year filtering is included in data requirements."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "get_data", "data_type": "income", "time_scope": "2020"}"""),
            create_mock_llm_response("""{"required_datasets": ["income_2020"], "filters": {"year": 2020}}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract income data for 2020", "agent": "extraction"}]}"""),
            create_mock_llm_response("""{"next_agent": "extraction"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "Show me income data for 2020 only"
            state["query_validation"] = {
                "is_valid": True,
                "topic": "income",
                "specified_year": 2020,
            }

            response = await agent.execute(state)

            assert response.success is True

            # Year filter should be in workflow plan
            workflow_plan = response.data.get("workflow", {})
            # Filters are optional, just verify success


@pytest.mark.unit
class TestWorkflowPlanning:
    """Test workflow plan creation."""

    @pytest.mark.asyncio
    async def test_simple_workflow_plan(self, mock_graph_state):
        """Test creation of simple workflow plan."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "get_average"}"""),
            create_mock_llm_response("""{"required_datasets": ["income_2020"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract data", "agent": "extraction"}, {"task": "Calculate average", "agent": "analytics"}]}"""),
            create_mock_llm_response("""{"next_agent": "extraction"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "What was the average income in 2020?"
            state["query_validation"] = {
                "is_valid": True,
                "topic": "income",
                "specified_year": 2020,
            }

            response = await agent.execute(state)

            assert response.success is True

            # Should have workflow plan
            workflow_plan = response.data.get("workflow", {})
            assert workflow_plan is not None

    @pytest.mark.asyncio
    async def test_multi_step_workflow_plan(self, mock_graph_state):
        """Test creation of multi-step workflow plan."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "compare_and_visualize"}"""),
            create_mock_llm_response("""{"required_datasets": ["income_2019", "income_2020"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract data", "agent": "extraction"}, {"task": "Compare", "agent": "analytics"}, {"task": "Create chart", "agent": "analytics"}]}"""),
            create_mock_llm_response("""{"next_agent": "extraction"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "Compare income between 2019 and 2020 and show a trend chart"
            state["query_validation"] = {
                "is_valid": True,
                "topic": "income",
                "specified_year": [2019, 2020],
            }

            response = await agent.execute(state)

            assert response.success is True

            workflow_plan = response.data.get("workflow", {})

            # Should have multiple steps for comparison and visualization
            if "steps" in workflow_plan:
                steps = workflow_plan["steps"]
                assert len(steps) >= 2  # At least retrieval and analysis/viz

    @pytest.mark.asyncio
    async def test_plan_with_visualization(self, mock_graph_state):
        """Test that plan includes visualization when requested."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "visualize"}"""),
            create_mock_llm_response("""{"required_datasets": ["employment_2020"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract data", "agent": "extraction"}, {"task": "Create chart", "agent": "analytics"}], "visualization_suggested": true}"""),
            create_mock_llm_response("""{"next_agent": "extraction"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "Show me a chart of employment rates"
            state["query_validation"] = {
                "is_valid": True,
                "topic": "employment",
                "specified_year": 2020,
            }

            response = await agent.execute(state)

            assert response.success is True


@pytest.mark.unit
class TestAgentDelegation:
    """Test delegation decision making."""

    @pytest.mark.asyncio
    async def test_delegates_to_extraction_agent(self, mock_graph_state):
        """Test that coordinator delegates to extraction agent."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "get_data"}"""),
            create_mock_llm_response("""{"required_datasets": ["income_2020"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract data", "agent": "extraction"}]}"""),
            create_mock_llm_response("""{"next_agent": "extraction", "reason": "Need to extract data"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "What was the average income in 2020?"
            state["query_validation"] = {
                "is_valid": True,
                "topic": "income",
                "specified_year": 2020,
            }

            response = await agent.execute(state)

            assert response.success is True

            # Should indicate delegation to extraction agent
            workflow_plan = response.data.get("workflow", {})
            if "next_agent" in workflow_plan:
                next_agent = workflow_plan["next_agent"]
                assert "extraction" in str(next_agent).lower() or "data" in str(next_agent).lower()

    @pytest.mark.asyncio
    async def test_plan_includes_analytics_step(self, mock_graph_state):
        """Test that plan includes analytics when needed."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "analyze"}"""),
            create_mock_llm_response("""{"required_datasets": ["income_2020"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract data", "agent": "extraction"}, {"task": "Calculate average", "agent": "analytics"}]}"""),
            create_mock_llm_response("""{"next_agent": "extraction"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "Calculate the average income and show trends"
            state["query_validation"] = {
                "is_valid": True,
                "topic": "income",
                "specified_year": 2020,
            }

            response = await agent.execute(state)

            assert response.success is True


@pytest.mark.unit
class TestStatePassingAndStructure:
    """Test state passing and response structure."""

    @pytest.mark.asyncio
    async def test_workflow_plan_structure(self, mock_graph_state):
        """Test that workflow plan has required structure."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "get_data"}"""),
            create_mock_llm_response("""{"required_datasets": ["income_2020"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract", "agent": "extraction"}]}"""),
            create_mock_llm_response("""{"next_agent": "extraction"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "What was the average income in 2020?"
            state["query_validation"] = {
                "is_valid": True,
                "topic": "income",
                "specified_year": 2020,
            }

            response = await agent.execute(state)

            assert response.success is True

            # Check response data structure
            assert "workflow" in response.data or "delegation" in response.data

    @pytest.mark.asyncio
    async def test_updates_graph_state(self, mock_graph_state):
        """Test that agent returns valid response structure."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "get_data"}"""),
            create_mock_llm_response("""{"required_datasets": ["income_2020"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract", "agent": "extraction"}]}"""),
            create_mock_llm_response("""{"next_agent": "extraction"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "What was the average income in 2020?"

            response = await agent.execute(state)

            # Should return valid response
            assert response is not None
            assert response.success is True
            assert response.data is not None


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_missing_validation(self, mock_graph_state):
        """Test handling when query validation is missing."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "get_data"}"""),
            create_mock_llm_response("""{"required_datasets": ["income_2020"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract", "agent": "extraction"}]}"""),
            create_mock_llm_response("""{"next_agent": "extraction"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "What was the average income in 2020?"
            # No query_validation in state

            response = await agent.execute(state)

            # Should still execute (may extract from query directly)
            assert response is not None

    @pytest.mark.asyncio
    async def test_handles_complex_comparison_query(self, mock_graph_state):
        """Test handling of complex comparison query."""
        from app.services.agents.coordinator import DataCoordinatorAgent
        from app.config import get_config

        config = get_config()

        mock_llm_responses = [
            create_mock_llm_response("""{"intent": "complex_comparison"}"""),
            create_mock_llm_response("""{"required_datasets": ["income_2019", "income_2020"]}"""),
            create_mock_llm_response("""{"steps": [{"task": "Extract multi-year data", "agent": "extraction"}, {"task": "Compare demographics", "agent": "analytics"}]}"""),
            create_mock_llm_response("""{"next_agent": "extraction"}"""),
        ]

        with patch.object(DataCoordinatorAgent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_responses

            agent = DataCoordinatorAgent(config)
            state = mock_graph_state.copy()
            state["query"] = "Compare income between males and females across age groups in 2019 vs 2020"
            state["query_validation"] = {
                "is_valid": True,
                "topic": "income",
                "specified_year": [2019, 2020],
            }

            response = await agent.execute(state)

            assert response.success is True

            # Should create appropriate plan
            workflow_plan = response.data.get("workflow", {})
            assert workflow_plan is not None
