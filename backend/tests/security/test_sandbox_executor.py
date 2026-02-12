"""Tests for sandbox code execution."""

import pandas as pd
import pytest

from app.services.security.sandbox_executor import SandboxExecutor


@pytest.fixture
def executor():
    """Create a sandbox executor instance using in-process fallback."""
    return SandboxExecutor(
        config={
            "use_process_isolation": False,  # Use fallback for tests
            "cpu_time_limit_seconds": 5,
            "memory_limit_mb": 512,
            "wall_time_limit_seconds": 10,
            "max_open_files": 3,
        }
    )


@pytest.fixture
def executor_no_isolation():
    """Create executor without process isolation (for testing fallback)."""
    return SandboxExecutor(config={"use_process_isolation": False})


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "year": [2020, 2021, 2022],
            "value": [100, 200, 300],
            "category": ["A", "B", "C"],
        }
    )


class TestSandboxExecutor:
    """Test suite for SandboxExecutor."""

    # ===== Test Successful Execution =====

    def test_execute_simple_code(self, executor, sample_dataframe):
        """Test execution of simple valid code."""
        code = "result = df['value'].sum()"
        result = executor.execute_code(code, context={"df": sample_dataframe})

        assert result.success
        assert result.result == 600
        assert result.error is None

    def test_execute_groupby_code(self, executor, sample_dataframe):
        """Test execution of groupby operation."""
        code = "result = df.groupby('category')['value'].sum()"
        result = executor.execute_code(code, context={"df": sample_dataframe})

        assert result.success
        assert result.result is not None
        assert isinstance(result.result, pd.Series)

    def test_execute_numpy_code(self, executor, sample_dataframe):
        """Test execution with numpy operations."""
        code = """
import numpy as np
result = np.mean(df['value'].values)
"""
        result = executor.execute_code(code, context={"df": sample_dataframe})

        assert result.success
        assert result.result == 200.0

    def test_execution_time_tracked(self, executor):
        """Test that execution time is tracked."""
        code = """
import time
time.sleep(0.1)
result = 42
"""
        result = executor.execute_code(code)

        assert result.success
        assert result.execution_time is not None
        assert result.execution_time >= 0.1

    # ===== Test Error Handling =====

    def test_handles_runtime_error(self, executor, sample_dataframe):
        """Test that runtime errors are caught and reported."""
        code = "result = df['nonexistent_column'].sum()"
        result = executor.execute_code(code, context={"df": sample_dataframe})

        assert not result.success
        assert result.error is not None
        assert result.error_type == "KeyError"

    def test_handles_zero_division(self, executor):
        """Test handling of ZeroDivisionError."""
        code = "result = 1 / 0"
        result = executor.execute_code(code)

        assert not result.success
        assert result.error_type == "ZeroDivisionError"

    def test_handles_type_error(self, executor):
        """Test handling of TypeError."""
        code = "result = 'string' + 123"
        result = executor.execute_code(code)

        assert not result.success
        assert result.error_type == "TypeError"

    def test_includes_traceback(self, executor):
        """Test that error traceback is included in metadata."""
        code = """
def foo():
    return 1 / 0

result = foo()
"""
        result = executor.execute_code(code)

        assert not result.success
        assert result.metadata is not None
        assert "traceback" in result.metadata
        assert "foo" in result.metadata["traceback"]

    # ===== Test Security: Blocked Operations =====

    def test_cannot_access_filesystem(self, executor):
        """Test that file access is blocked in execution environment."""
        code = """
# Note: open() is not available in restricted builtins
# This should fail at execution time
result = __builtins__['open']('/etc/passwd')
"""
        result = executor.execute_code(code)

        # Should fail because __builtins__ is restricted
        assert not result.success

    def test_restricted_builtins(self, executor):
        """Test that dangerous builtins are not available."""
        code = """
# Try to access eval from builtins
result = __builtins__['eval']('2 + 2')
"""
        result = executor.execute_code(code)

        # Should fail because eval is not in restricted builtins
        assert not result.success

    # ===== Test Fallback Execution =====

    def test_fallback_execution_works(self, executor_no_isolation, sample_dataframe):
        """Test that in-process fallback execution works."""
        code = "result = df['value'].sum()"
        result = executor_no_isolation.execute_code(code, context={"df": sample_dataframe})

        assert result.success
        assert result.result == 600

    def test_fallback_handles_errors(self, executor_no_isolation):
        """Test that fallback execution handles errors."""
        code = "result = undefined_variable"
        result = executor_no_isolation.execute_code(code)

        assert not result.success
        assert result.error_type == "NameError"

    # ===== Test Context Passing =====

    def test_multiple_context_variables(self, executor):
        """Test passing multiple variables in context."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        code = """
result = df1['a'].sum() + df2['b'].sum()
"""
        result = executor.execute_code(code, context={"df1": df1, "df2": df2})

        assert result.success
        assert result.result == 21  # (1+2+3) + (4+5+6)

    def test_empty_context(self, executor):
        """Test execution with no context."""
        code = "result = 2 + 2"
        result = executor.execute_code(code)

        assert result.success
        assert result.result == 4

    # ===== Test Result Extraction =====

    def test_result_variable_extracted(self, executor):
        """Test that 'result' variable is extracted from execution."""
        code = """
x = 10
y = 20
result = x + y
z = 30  # This should be ignored
"""
        result = executor.execute_code(code)

        assert result.success
        assert result.result == 30

    def test_no_result_variable(self, executor):
        """Test execution when no 'result' variable is set."""
        code = """
x = 10
y = 20
z = x + y
"""
        result = executor.execute_code(code)

        assert result.success
        assert result.result is None  # No 'result' variable

    # ===== Test Complex Operations =====

    def test_complex_pandas_operations(self, executor):
        """Test complex pandas code execution."""
        df = pd.DataFrame(
            {
                "year": [2020, 2021, 2022, 2020, 2021, 2022],
                "category": ["A", "A", "A", "B", "B", "B"],
                "value": [100, 150, 200, 120, 180, 240],
            }
        )

        code = """
import numpy as np

# Group by year and category
grouped = df.groupby(['year', 'category'])['value'].sum()

# Calculate mean across years
result = grouped.groupby(level='category').mean()
"""
        result = executor.execute_code(code, context={"df": df})

        assert result.success
        assert result.result is not None

    # ===== Test Edge Cases =====

    def test_empty_code(self, executor):
        """Test execution of empty code."""
        code = ""
        result = executor.execute_code(code)

        assert result.success
        assert result.result is None

    def test_code_with_only_comments(self, executor):
        """Test execution of code with only comments."""
        code = """
# This is a comment
# Another comment
"""
        result = executor.execute_code(code)

        assert result.success
        assert result.result is None

    def test_print_statements(self, executor):
        """Test that print statements work (output is discarded)."""
        code = """
print("Hello, world!")
result = 42
"""
        result = executor.execute_code(code)

        assert result.success
        assert result.result == 42
        # Note: print output is not captured in current implementation
