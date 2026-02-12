"""Tests for code validation security."""

import pandas as pd
import pytest

from app.services.security.code_validator import CodeValidator
from app.services.security.exceptions import CodeValidationError


class TestCodeValidator:
    """Test suite for CodeValidator."""

    @pytest.fixture
    def validator(self):
        """Create a code validator instance."""
        return CodeValidator(config={"use_ast_visitor": True})

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "year": [2020, 2021, 2022],
                "value": [100, 200, 300],
                "category": ["A", "B", "C"],
            }
        )

    # ===== Test Legitimate Code =====

    def test_valid_simple_code(self, validator):
        """Test that simple valid code passes validation."""
        code = """
result = df.groupby('year')['value'].sum()
"""
        result = validator.validate(code)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_valid_pandas_operations(self, validator):
        """Test legitimate pandas operations."""
        code = """
result = df.groupby('category')['value'].agg(['sum', 'mean', 'count'])
"""
        result = validator.validate(code)
        assert result.is_valid

    def test_valid_numpy_operations(self, validator):
        """Test legitimate numpy operations."""
        code = """
import numpy as np
result = np.mean(df['value'])
"""
        result = validator.validate(code)
        assert result.is_valid

    def test_valid_matplotlib_import(self, validator):
        """Test that matplotlib imports are allowed."""
        code = """
import matplotlib.pyplot as plt
result = df.plot(kind='bar')
"""
        result = validator.validate(code)
        assert result.is_valid

    def test_valid_datetime_operations(self, validator):
        """Test legitimate datetime operations."""
        code = """
from datetime import datetime
result = datetime.now()
"""
        result = validator.validate(code)
        assert result.is_valid

    # ===== Test Forbidden Attribute Access =====

    def test_blocks_globals_access(self, validator):
        """Test that __globals__ access is blocked."""
        code = "result = df.__class__.__init__.__globals__['__builtins__']"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("__globals__" in error for error in result.errors)

    def test_blocks_builtins_access(self, validator):
        """Test that __builtins__ access is blocked."""
        code = "result = pd.DataFrame.__init__.__globals__['__builtins__']"
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_class_access(self, validator):
        """Test that __class__ access is blocked."""
        code = "result = df.__class__.__name__"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("__class__" in error for error in result.errors)

    def test_blocks_dict_access(self, validator):
        """Test that __dict__ access is blocked."""
        code = "result = df.__dict__"
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_init_access(self, validator):
        """Test that __init__ access is blocked."""
        code = "result = df.__init__"
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_code_access(self, validator):
        """Test that __code__ access is blocked."""
        code = """
def foo():
    pass
result = foo.__code__
"""
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_subscript_dunder_access(self, validator):
        """Test that dunder access via subscript is blocked."""
        code = "result = df['__class__']"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("subscript" in error.lower() for error in result.errors)

    # ===== Test Forbidden Functions =====

    def test_blocks_eval(self, validator):
        """Test that eval() is blocked."""
        code = "result = eval('2 + 2')"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("eval" in error for error in result.errors)

    def test_blocks_exec(self, validator):
        """Test that exec() is blocked."""
        code = "exec('print(\"hello\")')"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("exec" in error for error in result.errors)

    def test_blocks_compile(self, validator):
        """Test that compile() is blocked."""
        code = "result = compile('2 + 2', '<string>', 'eval')"
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_getattr(self, validator):
        """Test that getattr() is blocked."""
        code = "result = getattr(df, '__class__')"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("getattr" in error for error in result.errors)

    def test_blocks_setattr(self, validator):
        """Test that setattr() is blocked."""
        code = "setattr(df, 'foo', 'bar')"
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_delattr(self, validator):
        """Test that delattr() is blocked."""
        code = "delattr(df, 'columns')"
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_import_builtin(self, validator):
        """Test that __import__ is blocked."""
        code = "result = __import__('os')"
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_open(self, validator):
        """Test that open() is blocked."""
        code = "result = open('/etc/passwd')"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("open" in error for error in result.errors)

    # ===== Test Database Operations =====

    def test_blocks_read_sql(self, validator):
        """Test that pd.read_sql() is blocked."""
        code = "result = pd.read_sql('SELECT * FROM users', conn)"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("read_sql" in error for error in result.errors)

    def test_blocks_read_sql_query(self, validator):
        """Test that pd.read_sql_query() is blocked."""
        code = "result = pd.read_sql_query('SELECT * FROM users', conn)"
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_read_sql_table(self, validator):
        """Test that pd.read_sql_table() is blocked."""
        code = "result = pd.read_sql_table('users', conn)"
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_to_sql(self, validator):
        """Test that df.to_sql() is blocked."""
        code = "df.to_sql('users', conn, if_exists='replace')"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("to_sql" in error for error in result.errors)

    # ===== Test Forbidden Imports =====

    def test_blocks_os_import(self, validator):
        """Test that os module import is blocked."""
        code = """
import os
result = os.listdir('.')
"""
        result = validator.validate(code)
        assert not result.is_valid
        assert any("os" in error for error in result.errors)

    def test_blocks_sys_import(self, validator):
        """Test that sys module import is blocked."""
        code = """
import sys
result = sys.version
"""
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_subprocess_import(self, validator):
        """Test that subprocess module is blocked."""
        code = """
import subprocess
result = subprocess.run(['ls'])
"""
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_socket_import(self, validator):
        """Test that socket module is blocked."""
        code = """
import socket
result = socket.gethostname()
"""
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_requests_import(self, validator):
        """Test that requests module is blocked."""
        code = """
import requests
result = requests.get('http://example.com')
"""
        result = validator.validate(code)
        assert not result.is_valid

    def test_blocks_sqlalchemy_import(self, validator):
        """Test that sqlalchemy import is blocked."""
        code = """
from sqlalchemy import create_engine
engine = create_engine('sqlite:///test.db')
"""
        result = validator.validate(code)
        assert not result.is_valid

    # ===== Test Syntax Errors =====

    def test_detects_syntax_error(self, validator):
        """Test that syntax errors are detected."""
        code = "result = df.groupby('year')['value'].sum("  # Missing closing paren
        result = validator.validate(code)
        assert not result.is_valid
        assert any("syntax" in error.lower() for error in result.errors)

    def test_detects_indentation_error(self, validator):
        """Test that indentation errors are detected."""
        code = """
result = df.groupby('year')
    .sum()  # Bad indentation
"""
        result = validator.validate(code)
        assert not result.is_valid

    # ===== Test Validation Helpers =====

    def test_validate_and_raise_success(self, validator):
        """Test that validate_and_raise doesn't raise on valid code."""
        code = "result = df.groupby('year')['value'].sum()"
        # Should not raise
        validator.validate_and_raise(code)

    def test_validate_and_raise_failure(self, validator):
        """Test that validate_and_raise raises on invalid code."""
        code = "result = eval('malicious')"
        with pytest.raises(CodeValidationError) as exc_info:
            validator.validate_and_raise(code)
        assert "eval" in str(exc_info.value)

    def test_column_validation_warning(self, validator, sample_dataframe):
        """Test that invalid column references generate warnings."""
        code = "result = df['nonexistent_column'].sum()"
        result = validator.validate(code, dataframe=sample_dataframe)
        assert result.is_valid  # Still valid, just a warning
        assert len(result.warnings) > 0
        assert any("nonexistent_column" in warning for warning in result.warnings)

    def test_column_validation_success(self, validator, sample_dataframe):
        """Test that valid column references don't generate warnings."""
        code = "result = df['year'].sum()"
        result = validator.validate(code, dataframe=sample_dataframe)
        assert result.is_valid
        # Should not have column-related warnings
        assert not any("not found" in warning for warning in result.warnings)

    # ===== Test Nested Attribute Access =====

    def test_warns_deeply_nested_attributes(self, validator):
        """Test that deeply nested attribute access triggers a warning."""
        code = "result = df.columns.values.tolist().copy().__class__"
        result = validator.validate(code)
        # Should fail because of __class__, but also warn about nesting
        assert not result.is_valid

    # ===== Test Configuration =====

    def test_disable_ast_visitor(self):
        """Test that AST visitor can be disabled."""
        validator = CodeValidator(config={"use_ast_visitor": False})
        # With AST disabled, dangerous code might pass (not recommended)
        code = "result = eval('2+2')"
        result = validator.validate(code)
        # Without AST validation, only syntax check happens
        assert result.is_valid  # Syntax is fine, but code is dangerous

    # ===== Test Edge Cases =====

    def test_empty_code(self, validator):
        """Test validation of empty code."""
        code = ""
        result = validator.validate(code)
        assert result.is_valid  # Empty code is technically valid

    def test_comment_only_code(self, validator):
        """Test validation of code with only comments."""
        code = "# This is just a comment"
        result = validator.validate(code)
        assert result.is_valid

    def test_multiline_code(self, validator):
        """Test validation of multiline code."""
        code = """
# Calculate summary statistics
grouped = df.groupby('category')
result = grouped['value'].agg(['sum', 'mean', 'std'])
"""
        result = validator.validate(code)
        assert result.is_valid

    def test_complex_legitimate_code(self, validator):
        """Test complex but legitimate pandas code."""
        code = """
import numpy as np

# Data transformation
df_filtered = df[df['value'] > 100]
df_sorted = df_filtered.sort_values('year')

# Aggregation
result = df_sorted.groupby('category').agg({
    'value': ['sum', 'mean', 'count'],
    'year': ['min', 'max']
})
"""
        result = validator.validate(code)
        assert result.is_valid
