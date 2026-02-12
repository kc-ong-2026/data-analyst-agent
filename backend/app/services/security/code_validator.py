"""AST-based code validation for secure code execution."""

import ast
import re
from dataclasses import dataclass, field

import pandas as pd

from .exceptions import CodeValidationError


@dataclass
class ValidationResult:
    """Result of code validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)

    def add_error(self, message: str, violation_type: str = None, node: ast.AST = None):
        """Add a validation error."""
        self.is_valid = False
        self.errors.append(message)
        violation = {"type": violation_type or "unknown", "message": message}
        if node:
            violation["line"] = getattr(node, "lineno", None)
            violation["col"] = getattr(node, "col_offset", None)
        self.violations.append(violation)

    def add_warning(self, message: str):
        """Add a validation warning."""
        self.warnings.append(message)


class SecurityASTVisitor(ast.NodeVisitor):
    """AST visitor that detects security violations."""

    # All dunder (double underscore) attributes that could be used for escaping
    FORBIDDEN_DUNDER_ATTRIBUTES = {
        "__globals__",
        "__builtins__",
        "__class__",
        "__dict__",
        "__init__",
        "__code__",
        "__closure__",
        "__func__",
        "__self__",
        "__module__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__import__",
        "__loader__",
        "__spec__",
        "__cached__",
        "__file__",
        "__name__",
        "__package__",
    }

    # Functions that can bypass security restrictions
    FORBIDDEN_FUNCTIONS = {
        "eval",
        "exec",
        "compile",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "help",
        "input",
        "open",
        "exit",
        "quit",
    }

    # Database operations that must be blocked
    FORBIDDEN_DATABASE_FUNCTIONS = {
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "to_sql",
        "execute",
        "executemany",
        "cursor",
        "connect",
    }

    # Allowed imports (whitelist)
    ALLOWED_IMPORTS = {
        "pandas",
        "numpy",
        "matplotlib",
        "matplotlib.pyplot",
        "datetime",
        "time",
        "math",
        "statistics",
        "collections",
    }

    def __init__(self):
        self.result = ValidationResult(is_valid=True)
        self.import_names: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        """Check import statements."""
        for alias in node.names:
            module_name = alias.name
            if not self._is_import_allowed(module_name):
                self.result.add_error(
                    f"Import not allowed: {module_name}",
                    violation_type="forbidden_import",
                    node=node,
                )
            else:
                self.import_names.add(alias.asname or module_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        """Check from...import statements."""
        module_name = node.module or ""
        if not self._is_import_allowed(module_name):
            self.result.add_error(
                f"Import not allowed: from {module_name}",
                violation_type="forbidden_import",
                node=node,
            )
        else:
            for alias in node.names:
                self.import_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        """Check attribute access."""
        # Check for forbidden dunder attributes
        if node.attr in self.FORBIDDEN_DUNDER_ATTRIBUTES:
            self.result.add_error(
                f"Access to forbidden attribute: {node.attr}",
                violation_type="forbidden_attribute",
                node=node,
            )

        # Check for nested attribute chains that might be suspicious
        if self._is_deeply_nested_attribute(node):
            self.result.add_warning(
                f"Deeply nested attribute access detected at line {node.lineno}"
            )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Check function calls."""
        func_name = self._get_function_name(node.func)

        # Check for forbidden functions
        if func_name in self.FORBIDDEN_FUNCTIONS:
            self.result.add_error(
                f"Forbidden function call: {func_name}",
                violation_type="forbidden_function",
                node=node,
            )

        # Check for database operations (e.g., pd.read_sql, df.to_sql)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in self.FORBIDDEN_DATABASE_FUNCTIONS:
                self.result.add_error(
                    f"Database operation not allowed: {node.func.attr}",
                    violation_type="forbidden_database_operation",
                    node=node,
                )

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        """Check subscript operations (could be used for dynamic attribute access)."""
        # Check for suspicious patterns like obj['__globals__']
        if isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                if node.slice.value in self.FORBIDDEN_DUNDER_ATTRIBUTES:
                    self.result.add_error(
                        f"Attempt to access forbidden attribute via subscript: {node.slice.value}",
                        violation_type="forbidden_attribute_subscript",
                        node=node,
                    )
        self.generic_visit(node)

    def _is_import_allowed(self, module_name: str) -> bool:
        """Check if import is in whitelist."""
        # Check exact match
        if module_name in self.ALLOWED_IMPORTS:
            return True
        # Check if it's a submodule of an allowed import
        return any(module_name.startswith(f"{allowed}.") for allowed in self.ALLOWED_IMPORTS)

    def _get_function_name(self, node: ast.AST) -> str:
        """Extract function name from call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _is_deeply_nested_attribute(self, node: ast.Attribute, depth: int = 0) -> bool:
        """Check if attribute access is deeply nested (potential obfuscation)."""
        if depth > 3:  # More than 3 levels of nesting
            return True
        if isinstance(node.value, ast.Attribute):
            return self._is_deeply_nested_attribute(node.value, depth + 1)
        return False


class CodeValidator:
    """Validates Python code for security before execution."""

    def __init__(self, config: dict = None):
        """
        Initialize code validator.

        Args:
            config: Optional configuration dict with security settings
        """
        self.config = config or {}
        self.use_ast_visitor = self.config.get("use_ast_visitor", True)
        self.block_dunder_attributes = self.config.get("block_dunder_attributes", True)

    def validate(self, code: str, dataframe: pd.DataFrame = None) -> ValidationResult:
        """
        Validate code for security violations.

        Args:
            code: Python code to validate
            dataframe: Optional DataFrame to validate column references against

        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        result = ValidationResult(is_valid=True)

        # Step 1: Validate syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result.add_error(
                f"Syntax error: {e.msg} at line {e.lineno}",
                violation_type="syntax_error",
            )
            return result

        # Step 2: AST-based security validation
        if self.use_ast_visitor:
            visitor = SecurityASTVisitor()
            visitor.visit(tree)
            result = visitor.result

        # Step 3: Validate DataFrame column references (if DataFrame provided)
        if dataframe is not None and result.is_valid:
            column_result = self._validate_dataframe_columns(code, dataframe)
            result.warnings.extend(column_result.warnings)

        return result

    def validate_and_raise(self, code: str, dataframe: pd.DataFrame = None) -> None:
        """
        Validate code and raise exception if validation fails.

        Args:
            code: Python code to validate
            dataframe: Optional DataFrame to validate column references against

        Raises:
            CodeValidationError: If validation fails
        """
        result = self.validate(code, dataframe)
        if not result.is_valid:
            raise CodeValidationError(
                f"Code validation failed: {'; '.join(result.errors)}",
                violation_type="validation_failed",
                details={
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "violations": result.violations,
                },
            )

    def _validate_dataframe_columns(self, code: str, dataframe: pd.DataFrame) -> ValidationResult:
        """
        Validate that column references in code exist in the DataFrame.

        This is a best-effort check using regex patterns.
        """
        result = ValidationResult(is_valid=True)
        available_columns = set(dataframe.columns)

        # Pattern to match column access: df['column'] or df["column"]
        column_patterns = [
            r"df\[['\"]([^'\"]+)['\"]\]",  # df['column']
            r"df\[([a-zA-Z_][a-zA-Z0-9_]*)\]",  # df[column_var]
        ]

        for pattern in column_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                column_name = match.group(1)
                if column_name not in available_columns:
                    result.add_warning(
                        f"Column '{column_name}' referenced in code but not found in DataFrame. "
                        f"Available columns: {', '.join(available_columns)}"
                    )

        return result
