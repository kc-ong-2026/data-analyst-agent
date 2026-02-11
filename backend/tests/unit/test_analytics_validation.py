"""Unit tests for Analytics Agent column validation."""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.agents.analytics.agent import AnalyticsAgent
from app.services.agents.base_agent import GraphState


@pytest.fixture
def analytics_agent():
    """Create analytics agent for testing."""
    return AnalyticsAgent()


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        "year": [2020, 2021, 2022],
        "sex": ["Male", "Female", "Male"],
        "age": ["25-34", "35-44", "25-34"],
        "employment_rate": [85.2, 78.3, 86.1]
    })


@pytest.fixture
def sample_state(sample_dataframe):
    """Create sample GraphState for testing."""
    return {
        "messages": [],
        "current_task": "What is the employment rate by age group?",
        "extracted_data": {
            "dataset1": {
                "source": "dataframe",
                "columns": ["year", "sex", "age", "employment_rate"],
                "dtypes": {"year": "int64", "sex": "object", "age": "object", "employment_rate": "float64"},
                "data": sample_dataframe.to_dict(orient="records"),
                "metadata": {
                    "summary_text": "Employment data by sex and age",
                    "primary_dimensions": ["sex", "age"],
                    "categorical_columns": ["sex", "age"],
                    "numeric_columns": ["employment_rate"],
                    "year_range": "2020-2022"
                }
            }
        },
        "analysis_results": {},
        "workflow_plan": [],
        "current_step": 0,
        "errors": [],
        "metadata": {},
        "intermediate_results": {
            "dataframes": {"dataset1": sample_dataframe},
            "data_summary": {
                "dataset1": {
                    "row_count": 3,
                    "columns": ["year", "sex", "age", "employment_rate"],
                    "numeric_columns": ["employment_rate"],
                    "shape": (3, 4),
                    "metadata": {
                        "summary_text": "Employment data by sex and age",
                        "primary_dimensions": ["sex", "age"],
                        "categorical_columns": ["sex", "age"],
                        "numeric_columns": ["employment_rate"],
                        "year_range": "2020-2022"
                    }
                }
            },
            "has_data": True
        },
        "should_continue": True,
        "retrieval_context": {},
        "query_validation": {},
        "available_years": {}
    }


class TestColumnValidation:
    """Tests for column validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_columns_exact_match(self, analytics_agent, sample_state):
        """Test validation when all required columns are present."""
        # Mock LLM response for exact match
        mock_response = """{
            "status": "exact_match",
            "reasoning": "All required columns present",
            "missing_concepts": [],
            "available_alternatives": ["sex", "age", "employment_rate"],
            "recommendation": "Proceed with code generation"
        }"""

        with patch.object(analytics_agent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await analytics_agent._validate_columns_node(sample_state)

            validation = result["intermediate_results"]["column_validation"]
            assert validation["status"] == "exact_match"
            assert len(validation["missing_concepts"]) == 0

    @pytest.mark.asyncio
    async def test_validate_columns_no_match(self, analytics_agent, sample_state):
        """Test validation when requested columns are missing."""
        # Update query to ask for non-existent data
        sample_state["current_task"] = "What is the employment rate in the technology sector?"

        # Mock LLM response for no match
        mock_response = """{
            "status": "no_match",
            "reasoning": "Dataset does not have sector breakdown",
            "missing_concepts": ["technology sector", "sector"],
            "available_alternatives": ["sex", "age"],
            "recommendation": "Explain what data is available"
        }"""

        with patch.object(analytics_agent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await analytics_agent._validate_columns_node(sample_state)

            validation = result["intermediate_results"]["column_validation"]
            assert validation["status"] == "no_match"
            assert "sector" in str(validation["missing_concepts"]).lower()

    @pytest.mark.asyncio
    async def test_validate_columns_partial_match(self, analytics_agent, sample_state):
        """Test validation with partial data availability."""
        # Mock LLM response for partial match
        mock_response = """{
            "status": "partial_match",
            "reasoning": "Have employment data but not for requested time period",
            "missing_concepts": ["data for 1990-2000"],
            "available_alternatives": ["data for 2020-2022"],
            "recommendation": "Generate code with caveat about time period"
        }"""

        with patch.object(analytics_agent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await analytics_agent._validate_columns_node(sample_state)

            validation = result["intermediate_results"]["column_validation"]
            assert validation["status"] == "partial_match"
            assert len(validation["available_alternatives"]) > 0

    @pytest.mark.asyncio
    async def test_fallback_response_generation(self, analytics_agent, sample_state):
        """Test that helpful fallback message is created when validation fails."""
        # Set validation result to no_match
        sample_state["intermediate_results"]["column_validation"] = {
            "status": "no_match",
            "reasoning": "Dataset doesn't have sector data",
            "missing_concepts": ["technology sector"],
            "available_alternatives": ["sex", "age"],
            "recommendation": "Suggest alternatives"
        }

        mock_fallback = "The dataset doesn't have employment data by sector. However, I can show employment rates by sex and age for 2020-2022."

        with patch.object(analytics_agent, '_invoke_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_fallback

            result = await analytics_agent._compose_fallback_response_node(sample_state)

            response_text = result["analysis_results"]["text"]
            assert "sector" in response_text.lower() or "sex" in response_text.lower()
            assert result["analysis_results"]["visualization"] is None

    def test_should_generate_code_routing(self, analytics_agent, sample_state):
        """Test routing logic based on validation status."""
        # Test exact_match -> generate
        sample_state["intermediate_results"]["column_validation"] = {"status": "exact_match"}
        assert analytics_agent._should_generate_code(sample_state) == "generate"

        # Test partial_match -> generate
        sample_state["intermediate_results"]["column_validation"] = {"status": "partial_match"}
        assert analytics_agent._should_generate_code(sample_state) == "generate"

        # Test no_match -> fallback
        sample_state["intermediate_results"]["column_validation"] = {"status": "no_match"}
        assert analytics_agent._should_generate_code(sample_state) == "fallback"


class TestHelperMethods:
    """Tests for helper methods."""

    def test_extract_json_from_response(self, analytics_agent):
        """Test JSON extraction from various response formats."""
        # Test JSON in code block
        response1 = '```json\n{"status": "exact_match"}\n```'
        result1 = analytics_agent._extract_json_from_response(response1)
        assert "exact_match" in result1

        # Test raw JSON
        response2 = 'Some text {"status": "no_match"} more text'
        result2 = analytics_agent._extract_json_from_response(response2)
        assert "no_match" in result2

    def test_extract_reasoning(self, analytics_agent):
        """Test reasoning extraction from LLM response."""
        response_with_reasoning = """
        <reasoning>
        1. User is asking about employment by age
        2. Dataset has age column
        3. We can provide this data
        </reasoning>

        ```python
        result = df.groupby('age')['employment_rate'].mean()
        ```
        """

        reasoning = analytics_agent._extract_reasoning(response_with_reasoning)
        assert "age" in reasoning.lower()
        assert "employment" in reasoning.lower()

        # Test with no reasoning tags
        response_no_reasoning = "Some response without reasoning tags"
        reasoning_empty = analytics_agent._extract_reasoning(response_no_reasoning)
        assert "No explicit reasoning" in reasoning_empty

    def test_make_user_friendly_error_server_errors(self, analytics_agent):
        """Test that server errors are properly categorized."""
        # Test AttributeError with 'object has no attribute'
        server_error = AttributeError("'AgentState' object has no attribute 'get'")
        message = analytics_agent._make_user_friendly_error(server_error, "analytics execution")

        assert "internal server error" in message.lower()
        assert "our team has been notified" in message.lower()
        assert "Error ID" in message
        assert "'AgentState' object has no attribute 'get'" not in message or "Technical" in message

    def test_make_user_friendly_error_user_errors(self, analytics_agent):
        """Test that user-facing errors get helpful messages."""
        # Test KeyError (data issue)
        user_error = KeyError("column_name")
        message = analytics_agent._make_user_friendly_error(user_error, "data processing")

        assert "couldn't find" in message.lower()
        assert "expected data field" in message.lower()
        assert "internal server error" not in message.lower()

        # Test ValueError
        value_error = ValueError("Invalid value for column")
        message = analytics_agent._make_user_friendly_error(value_error, "data validation")

        assert "invalid data" in message.lower()
        assert "internal server error" not in message.lower()
