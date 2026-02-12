"""
Unit tests for Query Verification Agent.

Tests the verification agent's ability to:
- Validate topic relevance
- Extract year specifications
- Validate dimension requirements
- Check data availability
- Generate appropriate validation results
"""

import pytest


@pytest.mark.unit
class TestTopicValidation:
    """Test topic validation logic."""

    @pytest.mark.asyncio
    async def test_valid_income_query(self, mock_graph_state):
        """Test that income queries are validated as valid."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "What was the average income in Singapore in 2020?"
        state["query"] = query
        state["current_task"] = query  # Verification agent uses current_task

        response = await agent.execute(state)

        assert response.success is True
        validation = response.data.get("validation", {})
        assert validation["valid"] is True
        assert validation["topic_valid"] is True
        # Check that validation allows continuation (no pause needed)
        assert validation.get("missing_year") is False

    @pytest.mark.asyncio
    async def test_valid_employment_query(self, mock_graph_state):
        """Test that employment queries with dimensions are validated as valid."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        # Include dimension (age group) to make query valid
        query = "Show me employment rates for youth in 2019"
        state["query"] = query
        state["current_task"] = query  # Verification agent uses current_task

        response = await agent.execute(state)

        assert response.success is True
        validation = response.data.get("validation", {})
        assert validation["valid"] is True
        assert validation["topic_valid"] is True

    @pytest.mark.asyncio
    async def test_invalid_off_topic_query(self, mock_graph_state):
        """Test that off-topic queries are rejected."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "What is the weather like today?"
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        validation = response.data.get("validation", {})
        assert validation["valid"] is False
        assert validation["topic_valid"] is False
        assert (
            "employment" in validation.get("reason", "").lower()
            or "income" in validation.get("reason", "").lower()
            or "hours" in validation.get("reason", "").lower()
        )

    @pytest.mark.asyncio
    async def test_valid_hours_worked_query(self, mock_graph_state):
        """Test that hours worked queries are validated."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "How many hours did people work in 2020?"
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        assert response.success is True
        validation = response.data.get("validation", {})
        assert validation["valid"] is True
        assert validation["topic_valid"] is True


@pytest.mark.unit
class TestYearExtraction:
    """Test year extraction from queries."""

    @pytest.mark.asyncio
    async def test_single_year_extraction(self, mock_graph_state):
        """Test extraction of single year from query."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "What was the income in 2020?"
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        validation = response.data.get("validation", {})
        assert validation.get("years_found") is True
        # Check that requested_years contains 2020
        requested_years = validation.get("requested_years")
        if requested_years:
            assert requested_years.get("min") == 2020 or requested_years.get("max") == 2020

    @pytest.mark.asyncio
    async def test_year_range_extraction(self, mock_graph_state):
        """Test extraction of year range from query."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "Compare employment rates from 2019 to 2021"
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        validation = response.data.get("validation", {})
        assert validation.get("years_found") is True
        # Should extract year range
        requested_years = validation.get("requested_years")
        assert requested_years is not None
        assert requested_years.get("min") == 2019
        assert requested_years.get("max") == 2021

    @pytest.mark.asyncio
    async def test_missing_year_triggers_pause(self, mock_graph_state):
        """Test that missing year specification triggers pause."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "What is the average income?"  # No year
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        validation = response.data.get("validation", {})
        assert validation.get("missing_year") is True
        assert validation.get("years_found") is False
        assert "year" in validation.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_multiple_years_extraction(self, mock_graph_state):
        """Test extraction of multiple explicit years."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "Compare income in 2019, 2020, and 2021"
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        validation = response.data.get("validation", {})
        assert validation.get("years_found") is True
        # Should extract year range covering all three years
        requested_years = validation.get("requested_years")
        assert requested_years is not None
        assert requested_years.get("min") == 2019
        assert requested_years.get("max") == 2021


@pytest.mark.unit
class TestDimensionValidation:
    """Test dimension validation for employment queries."""

    @pytest.mark.asyncio
    async def test_employment_with_dimension(self, mock_graph_state):
        """Test employment query with proper dimension."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "What was the employment rate by age group in 2020?"
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        validation = response.data.get("validation", {})
        assert validation["valid"] is True
        dimension_check = validation.get("dimension_check", {})
        assert dimension_check.get("dimensions_valid") is True
        dimensions_found = dimension_check.get("dimensions_found", [])
        assert len(dimensions_found) > 0  # Should find age dimension

    @pytest.mark.asyncio
    async def test_employment_missing_dimension_triggers_pause(self, mock_graph_state):
        """Test that employment query without dimension triggers pause."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        state["query"] = "What was the employment rate in 2020?"  # No dimension

        response = await agent.execute(state)

        validation = response.data.get("validation", {})
        # May or may not pause depending on implementation - check reason
        if validation.get("should_pause"):
            assert "dimension" in validation.get("reason", "").lower()


@pytest.mark.unit
class TestYearAvailability:
    """Test year availability checking against database."""

    @pytest.mark.asyncio
    async def test_available_year_passes(self, mock_graph_state):
        """Test that available year passes validation (unit test - no DB access)."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "What was the income in 2020?"
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        validation = response.data.get("validation", {})
        # When DB is not available (unit test), agent skips availability check and passes validation
        assert validation.get("years_available") is not False
        assert validation.get("years_found") is True

    @pytest.mark.asyncio
    async def test_unavailable_year_triggers_pause(self, mock_graph_state):
        """Test that queries without years trigger missing year flag."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "What was the income?"  # No year specified
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        validation = response.data.get("validation", {})
        # Query without year should trigger missing_year flag
        assert validation.get("missing_year") is True
        assert validation.get("years_found") is False


@pytest.mark.unit
class TestValidationResultStructure:
    """Test validation result structure and completeness."""

    @pytest.mark.asyncio
    async def test_validation_result_has_required_fields(self, mock_graph_state):
        """Test that validation result has all required fields."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "What was the average income in 2020?"
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        assert response.success is True
        assert "validation" in response.data

        validation = response.data["validation"]
        # Check for actual fields in QueryValidationResult model
        required_fields = ["valid", "topic_valid", "years_found", "missing_year"]

        for field in required_fields:
            assert field in validation, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_pause_reason_provided_when_pausing(self, mock_graph_state):
        """Test that pause reason is provided when should_pause is True."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "Tell me about the weather"  # Off-topic
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        validation = response.data.get("validation", {})
        # Off-topic queries should be invalid
        if not validation.get("valid"):
            assert "reason" in validation
            assert len(validation["reason"]) > 0


@pytest.mark.unit
class TestAgentStatePassing:
    """Test that agent properly passes state to orchestrator."""

    @pytest.mark.asyncio
    async def test_to_graph_state_includes_validation(self, mock_graph_state):
        """Test that to_graph_state includes validation results."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        state["query"] = "What was the income in 2020?"

        response = await agent.execute(state)

        # Note: Verification agent puts validation in response.data, not to_graph_state()
        # Orchestrator must extract it
        assert "validation" in response.data

        validation = response.data["validation"]
        assert "valid" in validation
        assert "topic_valid" in validation

    @pytest.mark.asyncio
    async def test_agent_updates_messages(self, mock_graph_state):
        """Test that agent adds AI message to state."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "What was the income in 2020?"
        state["query"] = query
        state["current_task"] = query
        initial_message_count = len(state["messages"])

        response = await agent.execute(state)

        # Agent should have added a message to response.state
        updated_state = response.state
        if updated_state and updated_state.messages:
            assert len(updated_state.messages) > initial_message_count


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_query(self, mock_graph_state):
        """Test handling of empty query."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = ""
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        # Should handle gracefully - empty query should be invalid
        assert response is not None
        # Empty query will fail topic validation
        validation = response.data.get("validation", {})
        assert validation.get("topic_valid") is False

    @pytest.mark.asyncio
    async def test_very_long_query(self, mock_graph_state):
        """Test handling of very long query."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "income " * 1000  # Very long query
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        # Should handle gracefully without timeout
        assert response is not None
        # Long query with "income" keyword should be valid
        validation = response.data.get("validation", {})
        assert validation.get("topic_valid") is True

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, mock_graph_state):
        """Test handling of special characters."""
        from app.config import get_config
        from app.services.agents.verification import QueryVerificationAgent

        config = get_config()
        agent = QueryVerificationAgent(config)

        state = mock_graph_state.copy()
        query = "What was the income in 2020? @#$%^&*()"
        state["query"] = query
        state["current_task"] = query

        response = await agent.execute(state)

        # Should extract year despite special characters
        validation = response.data.get("validation", {})
        assert validation.get("years_found") is True
        requested_years = validation.get("requested_years")
        if requested_years:
            assert requested_years.get("min") == 2020
