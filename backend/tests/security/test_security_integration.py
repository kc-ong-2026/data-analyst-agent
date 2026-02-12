"""End-to-end security integration tests."""

import pandas as pd
import pytest

from app.services.security import (
    CodeExecutionAuditLogger,
    CodeValidator,
    SandboxExecutor,
)


@pytest.fixture
def security_config():
    """Create security configuration for testing."""
    return {
        "enabled": True,
        "validation": {
            "use_ast_visitor": True,
            "block_dunder_attributes": True,
        },
        "isolation": {
            "use_process_isolation": False,  # Use in-process fallback for tests
            "cpu_time_limit_seconds": 5,
            "memory_limit_mb": 512,
            "wall_time_limit_seconds": 10,
            "max_open_files": 3,
        },
        "audit": {
            "enabled": True,
            "log_file": "/tmp/test_code_execution_audit.log",
        },
    }


@pytest.fixture
def validator(security_config):
    """Create code validator."""
    return CodeValidator(config=security_config["validation"])


@pytest.fixture
def executor(security_config):
    """Create sandbox executor using in-process fallback."""
    return SandboxExecutor(config=security_config["isolation"])


@pytest.fixture
def audit_logger(security_config):
    """Create audit logger."""
    return CodeExecutionAuditLogger(config=security_config["audit"])


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame."""
    return pd.DataFrame(
        {
            "year": [2020, 2021, 2022],
            "value": [100, 200, 300],
            "category": ["A", "B", "C"],
        }
    )


class TestSecurityIntegration:
    """Integration tests for the complete security stack."""

    # ===== Test Complete Security Flow =====

    def test_validate_then_execute_legitimate_code(self, validator, executor, sample_dataframe):
        """Test full flow with legitimate code."""
        code = "result = df.groupby('category')['value'].sum()"

        # Step 1: Validate
        validation_result = validator.validate(code, dataframe=sample_dataframe)
        assert validation_result.is_valid

        # Step 2: Execute
        execution_result = executor.execute_code(code, context={"df": sample_dataframe})
        assert execution_result.success
        assert execution_result.result is not None

    def test_validate_blocks_dangerous_code(self, validator, executor):
        """Test that dangerous code is blocked at validation."""
        code = "result = eval('malicious code')"

        # Step 1: Validation should fail
        validation_result = validator.validate(code)
        assert not validation_result.is_valid
        assert any("eval" in error for error in validation_result.errors)

        # Should not proceed to execution, but if it does, sandbox should also block
        execution_result = executor.execute_code(code)
        # Even if validation is bypassed, execution should fail due to restricted builtins
        assert not execution_result.success

    def test_validate_blocks_attribute_escape(self, validator, executor, sample_dataframe):
        """Test that __globals__ access is blocked."""
        code = "result = df.__class__.__init__.__globals__['__builtins__']"

        # Validation should block it
        validation_result = validator.validate(code)
        assert not validation_result.is_valid
        assert any(
            "__class__" in error or "__init__" in error for error in validation_result.errors
        )

    def test_validate_blocks_database_operations(self, validator):
        """Test that database operations are blocked."""
        codes = [
            "result = pd.read_sql('SELECT * FROM users', conn)",
            "df.to_sql('users', conn, if_exists='replace')",
            "result = pd.read_sql_query('SELECT * FROM data', engine)",
        ]

        for code in codes:
            validation_result = validator.validate(code)
            assert not validation_result.is_valid
            assert any(
                "database" in error.lower() or "sql" in error.lower()
                for error in validation_result.errors
            )

    def test_audit_logging_captures_execution(
        self, validator, executor, audit_logger, sample_dataframe
    ):
        """Test that audit logger captures execution details."""
        code = "result = df['value'].sum()"
        query = "What is the total value?"

        # Validate
        validation_result = validator.validate(code, dataframe=sample_dataframe)

        # Execute
        execution_result = executor.execute_code(code, context={"df": sample_dataframe})

        # Audit log
        audit_logger.log_execution(
            code=code,
            query=query,
            validation_result=validation_result,
            execution_result=execution_result,
        )

        # Check that log file was created (basic check)
        import os

        assert os.path.exists("/tmp/test_code_execution_audit.log")

    def test_audit_logging_security_violation(self, audit_logger):
        """Test that security violations are logged."""
        code = "result = eval('2+2')"

        audit_logger.log_security_violation(
            code=code,
            violation_type="forbidden_function",
            details="Use of eval() function",
            severity="high",
        )

        # Verify log file exists
        import os

        assert os.path.exists("/tmp/test_code_execution_audit.log")

    # ===== Test Configuration Toggle =====

    def test_security_can_be_disabled(self):
        """Test that security can be disabled for testing."""
        validator = CodeValidator(config={"use_ast_visitor": False})

        # With AST disabled, dangerous code passes syntax check
        code = "result = eval('2+2')"
        validation_result = validator.validate(code)
        assert validation_result.is_valid  # Only syntax checked

    def test_process_isolation_can_be_disabled(self, sample_dataframe):
        """Test that process isolation can be disabled."""
        executor = SandboxExecutor(config={"use_process_isolation": False})

        code = "result = df['value'].sum()"
        execution_result = executor.execute_code(code, context={"df": sample_dataframe})

        assert execution_result.success
        assert execution_result.result == 600

    # ===== Test Complex Real-World Scenarios =====

    def test_complex_pandas_analysis(self, validator, executor):
        """Test complex but legitimate pandas code."""
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

# Calculate percentage growth
result = grouped.pct_change().dropna()
"""

        # Validate
        validation_result = validator.validate(code, dataframe=df)
        assert validation_result.is_valid

        # Execute
        execution_result = executor.execute_code(code, context={"df": df})
        assert execution_result.success

    def test_visualization_code(self, validator, executor, sample_dataframe):
        """Test matplotlib visualization code."""
        code = """
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
df.plot(kind='bar', x='category', y='value', ax=ax)
ax.set_title('Values by Category')
result = fig
"""

        # Validate
        validation_result = validator.validate(code, dataframe=sample_dataframe)
        assert validation_result.is_valid

        # Execute
        execution_result = executor.execute_code(code, context={"df": sample_dataframe})
        assert execution_result.success
        # Result should be a matplotlib Figure
        from matplotlib.figure import Figure

        assert isinstance(execution_result.result, Figure)

    # ===== Test Error Handling =====

    def test_handles_runtime_errors_gracefully(self, executor, sample_dataframe):
        """Test that runtime errors are caught and reported."""
        code = "result = df['nonexistent_column'].sum()"

        execution_result = executor.execute_code(code, context={"df": sample_dataframe})

        assert not execution_result.success
        assert execution_result.error_type == "KeyError"
        assert "nonexistent_column" in execution_result.error

    def test_handles_syntax_errors_in_validation(self, validator):
        """Test that syntax errors are caught in validation."""
        code = "result = df.groupby('year')['value'].sum("  # Missing closing paren

        validation_result = validator.validate(code)
        assert not validation_result.is_valid
        assert any("syntax" in error.lower() for error in validation_result.errors)


class TestSecurityDefenseInDepth:
    """Tests that verify defense-in-depth security."""

    def test_multiple_security_layers(self):
        """Test that multiple security layers are in place."""
        # Layer 1: AST Validation
        validator = CodeValidator(config={"use_ast_visitor": True})

        # Layer 2: Process Isolation (using fallback for tests)
        executor = SandboxExecutor(config={"use_process_isolation": False})

        # Layer 3: Audit Logging
        audit_logger = CodeExecutionAuditLogger(
            config={
                "enabled": True,
                "log_file": "/tmp/test_defense_in_depth.log",
            }
        )

        dangerous_code = "result = __import__('os').system('ls')"

        # Layer 1 should block
        validation_result = validator.validate(dangerous_code)
        assert not validation_result.is_valid

        # Log the violation
        audit_logger.log_security_violation(
            code=dangerous_code,
            violation_type="forbidden_import",
            details="Attempt to import os module",
            severity="critical",
        )

    def test_cannot_bypass_restricted_builtins(self):
        """Test that restricted builtins cannot be bypassed."""
        executor = SandboxExecutor(config={"use_process_isolation": False})

        # Try various bypass techniques
        bypass_attempts = [
            "__builtins__['eval']('2+2')",
            "__builtins__['__import__']('os')",
            "globals()['__builtins__']['open']('/etc/passwd')",
        ]

        for code in bypass_attempts:
            result = executor.execute_code(code)
            # Should fail because __builtins__ is restricted
            assert not result.success
