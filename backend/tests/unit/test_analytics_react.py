"""Unit tests for Analytics Agent ReAct pattern."""

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.services.agents.analytics.agent import AnalyticsAgent


@pytest.fixture
def analytics_agent():
    """Create analytics agent for testing."""
    return AnalyticsAgent()


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "year": [2020, 2021, 2022],
            "sex": ["Male", "Female", "Male"],
            "age": ["25-34", "35-44", "25-34"],
            "employment_rate": [85.2, 78.3, 86.1],
        }
    )


class TestCodeValidation:
    """Tests for code validation functionality."""

    def test_validate_code_syntax_error(self, analytics_agent, sample_dataframe):
        """Test detection of syntax errors."""
        bad_code = "result = df.groupby('age')['employment_rate'.mean()"  # Missing closing bracket

        validation = analytics_agent._validate_generated_code(
            code=bad_code, dataframes={"df": sample_dataframe}, should_plot=False
        )

        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
        assert "syntax" in validation["errors"][0].lower()

    def test_validate_code_missing_columns(self, analytics_agent, sample_dataframe):
        """Test detection of non-existent column references."""
        code_with_bad_col = "result = df.groupby('sector')['employment_rate'].mean()"

        validation = analytics_agent._validate_generated_code(
            code=code_with_bad_col, dataframes={"df": sample_dataframe}, should_plot=False
        )

        assert validation["valid"] is False
        assert any("sector" in str(err).lower() for err in validation["errors"])

    def test_validate_code_forbidden_imports(self, analytics_agent, sample_dataframe):
        """Test rejection of forbidden imports."""
        code_with_os = """
import os
result = df.head()
"""

        validation = analytics_agent._validate_generated_code(
            code=code_with_os, dataframes={"df": sample_dataframe}, should_plot=False
        )

        assert validation["valid"] is False
        assert any("forbidden" in str(err).lower() for err in validation["errors"])

    def test_validate_code_forbidden_operations(self, analytics_agent, sample_dataframe):
        """Test rejection of file I/O and eval/exec."""
        code_with_file_io = """
with open('file.txt', 'w') as f:
    f.write('data')
result = df.head()
"""

        validation = analytics_agent._validate_generated_code(
            code=code_with_file_io, dataframes={"df": sample_dataframe}, should_plot=False
        )

        assert validation["valid"] is False
        assert any("file" in str(err).lower() for err in validation["errors"])

    def test_validate_code_matplotlib_check(self, analytics_agent, sample_dataframe):
        """Test matplotlib requirement when visualization requested."""
        code_without_plot = "result = df.groupby('age')['employment_rate'].mean()"

        validation = analytics_agent._validate_generated_code(
            code=code_without_plot, dataframes={"df": sample_dataframe}, should_plot=True
        )

        # Should have warning about missing matplotlib
        assert len(validation["warnings"]) > 0
        assert any("matplotlib" in str(warn).lower() for warn in validation["warnings"])

    def test_validate_code_valid_code(self, analytics_agent, sample_dataframe):
        """Test that valid code passes validation."""
        valid_code = """
result = df.groupby('age')['employment_rate'].mean()
result = result.head(100)
"""

        validation = analytics_agent._validate_generated_code(
            code=valid_code, dataframes={"df": sample_dataframe}, should_plot=False
        )

        assert validation["valid"] is True
        assert len(validation["errors"]) == 0


class TestReactLoop:
    """Tests for ReAct loop functionality."""

    @pytest.mark.asyncio
    async def test_react_iteration_tracking(self, analytics_agent, sample_dataframe):
        """Test that ReAct iterations are tracked correctly."""
        state = {
            "messages": [],
            "current_task": "Show employment by age",
            "extracted_data": {},
            "analysis_results": {},
            "workflow_plan": [],
            "current_step": 0,
            "errors": [],
            "metadata": {},
            "intermediate_results": {
                "dataframes": {"df": sample_dataframe},
                "data_summary": {
                    "df": {
                        "row_count": 3,
                        "columns": ["year", "sex", "age", "employment_rate"],
                        "numeric_columns": ["employment_rate"],
                        "metadata": {},
                    }
                },
                "react_iteration": 0,
                "react_max_iterations": 3,
                "react_history": [],
                "react_feedback": None,
            },
            "should_continue": True,
            "retrieval_context": {},
            "query_validation": {},
            "available_years": {},
        }

        # Mock LLM to return code with reasoning
        mock_response = """
<reasoning>
I need to group by age and calculate mean employment rate.
</reasoning>

```python
result = df.groupby('age')['employment_rate'].mean()
```
"""

        with patch.object(analytics_agent, "_invoke_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await analytics_agent._generate_code_node(state)

            assert result["intermediate_results"]["react_iteration"] == 0
            assert len(result["intermediate_results"]["react_history"]) == 1
            assert "reasoning" in result["intermediate_results"]["react_history"][0]

    @pytest.mark.asyncio
    async def test_evaluate_results_success(self, analytics_agent):
        """Test evaluation recognizes successful execution."""
        state = {
            "messages": [],
            "current_task": "Show employment",
            "extracted_data": {},
            "analysis_results": {},
            "workflow_plan": [],
            "current_step": 0,
            "errors": [],
            "metadata": {},
            "intermediate_results": {
                "execution_result": pd.Series([85.2, 78.3]),
                "execution_error": None,
                "code_validation": {"valid": True},
                "should_plot": False,
                "react_iteration": 0,
                "react_max_iterations": 3,
                "react_history": [
                    {"iteration": 0, "reasoning": "test", "action": "code", "observation": None}
                ],
            },
            "should_continue": True,
            "retrieval_context": {},
            "query_validation": {},
            "available_years": {},
        }

        result = await analytics_agent._evaluate_results_node(state)

        evaluation = result["intermediate_results"]["evaluation"]
        assert evaluation["success"] is True
        assert evaluation["should_retry"] is False
        assert evaluation["feedback"] is None

    @pytest.mark.asyncio
    async def test_evaluate_results_failure_empty_dataframe(self, analytics_agent):
        """Test evaluation detects empty DataFrame result."""
        state = {
            "messages": [],
            "current_task": "Show employment",
            "extracted_data": {},
            "analysis_results": {},
            "workflow_plan": [],
            "current_step": 0,
            "errors": [],
            "metadata": {},
            "intermediate_results": {
                "execution_result": pd.DataFrame(),  # Empty DataFrame
                "execution_error": None,
                "code_validation": {"valid": True},
                "should_plot": False,
                "react_iteration": 0,
                "react_max_iterations": 3,
                "react_history": [
                    {"iteration": 0, "reasoning": "test", "action": "code", "observation": None}
                ],
            },
            "should_continue": True,
            "retrieval_context": {},
            "query_validation": {},
            "available_years": {},
        }

        result = await analytics_agent._evaluate_results_node(state)

        evaluation = result["intermediate_results"]["evaluation"]
        assert evaluation["success"] is False
        assert evaluation["should_retry"] is True
        assert "empty" in evaluation["feedback"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_results_failure_execution_error(self, analytics_agent):
        """Test evaluation detects execution errors."""
        state = {
            "messages": [],
            "current_task": "Show employment",
            "extracted_data": {},
            "analysis_results": {},
            "workflow_plan": [],
            "current_step": 0,
            "errors": [],
            "metadata": {},
            "intermediate_results": {
                "execution_result": None,
                "execution_error": "KeyError: 'sector'",
                "code_validation": {"valid": True},
                "should_plot": False,
                "react_iteration": 0,
                "react_max_iterations": 3,
                "react_history": [
                    {"iteration": 0, "reasoning": "test", "action": "code", "observation": None}
                ],
            },
            "should_continue": True,
            "retrieval_context": {},
            "query_validation": {},
            "available_years": {},
        }

        result = await analytics_agent._evaluate_results_node(state)

        evaluation = result["intermediate_results"]["evaluation"]
        assert evaluation["success"] is False
        assert evaluation["should_retry"] is True
        assert "execution error" in evaluation["feedback"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_results_max_iterations(self, analytics_agent):
        """Test that retry stops after max iterations."""
        state = {
            "messages": [],
            "current_task": "Show employment",
            "extracted_data": {},
            "analysis_results": {},
            "workflow_plan": [],
            "current_step": 0,
            "errors": [],
            "metadata": {},
            "intermediate_results": {
                "execution_result": None,
                "execution_error": "Some error",
                "code_validation": {"valid": True},
                "should_plot": False,
                "react_iteration": 2,  # Already at iteration 2 (0-indexed, so this is 3rd iteration)
                "react_max_iterations": 3,
                "react_history": [],
            },
            "should_continue": True,
            "retrieval_context": {},
            "query_validation": {},
            "available_years": {},
        }

        result = await analytics_agent._evaluate_results_node(state)

        evaluation = result["intermediate_results"]["evaluation"]
        assert evaluation["should_retry"] is False  # Should not retry after max iterations

    def test_should_retry_generation_routing(self, analytics_agent):
        """Test routing logic for ReAct loop."""
        # Test retry path
        state_retry = {"intermediate_results": {"evaluation": {"should_retry": True}}}
        assert analytics_agent._should_retry_generation(state_retry) == "retry"

        # Test continue path
        state_continue = {"intermediate_results": {"evaluation": {"should_retry": False}}}
        assert analytics_agent._should_retry_generation(state_continue) == "continue"


class TestReactPrompts:
    """Tests for ReAct prompt generation."""

    def test_build_react_prompt_first_iteration(self, analytics_agent):
        """Test prompt for first iteration (no feedback)."""
        prompt = analytics_agent._build_react_prompt(
            query="Show employment by age",
            columns_str="year, age, employment_rate",
            dtypes_str="year: int64\nage: object\nemployment_rate: float64",
            sample_rows=[{"year": 2020, "age": "25-34", "employment_rate": 85.2}],
            summary_text="Employment data",
            should_plot=False,
            validation_context=None,
            iteration=0,
            feedback=None,
            history=[],
        )

        assert "ReAct Iteration 1 of 3" in prompt
        assert "<reasoning>" in prompt
        assert "```python" in prompt
        assert "Previous Attempt Failed" not in prompt

    def test_build_react_prompt_with_feedback(self, analytics_agent):
        """Test prompt includes feedback from previous iteration."""
        prompt = analytics_agent._build_react_prompt(
            query="Show employment by age",
            columns_str="year, age, employment_rate",
            dtypes_str="year: int64\nage: object\nemployment_rate: float64",
            sample_rows=[{"year": 2020, "age": "25-34", "employment_rate": 85.2}],
            summary_text="Employment data",
            should_plot=False,
            validation_context=None,
            iteration=1,
            feedback="Execution error: KeyError 'sector'",
            history=[
                {
                    "iteration": 0,
                    "reasoning": "Group by sector",
                    "action": "df.groupby('sector')",
                    "observation": {"error": "KeyError: 'sector'"},
                }
            ],
        )

        assert "ReAct Iteration 2 of 3" in prompt
        assert "Previous Attempt Failed" in prompt
        assert "KeyError" in prompt
        assert "Iteration 1" in prompt  # Shows history

    def test_build_react_prompt_with_validation_context(self, analytics_agent):
        """Test prompt includes validation context for partial matches."""
        validation_context = {
            "status": "partial_match",
            "missing_concepts": ["sector"],
            "available_alternatives": ["age", "sex"],
        }

        prompt = analytics_agent._build_react_prompt(
            query="Show employment by sector",
            columns_str="year, age, sex, employment_rate",
            dtypes_str="year: int64\nage: object\nsex: object\nemployment_rate: float64",
            sample_rows=[],
            summary_text="Employment data",
            should_plot=False,
            validation_context=validation_context,
            iteration=0,
            feedback=None,
            history=[],
        )

        assert "DATA AVAILABILITY NOTICE" in prompt
        assert "sector" in prompt
        assert "age" in prompt or "sex" in prompt
