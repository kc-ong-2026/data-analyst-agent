"""Security-related exceptions for code execution."""


class SecurityError(Exception):
    """Base exception for security violations."""

    pass


class CodeValidationError(SecurityError):
    """Exception raised when code validation fails."""

    def __init__(self, message: str, violation_type: str = None, details: dict = None):
        super().__init__(message)
        self.violation_type = violation_type
        self.details = details or {}


class SandboxExecutionError(SecurityError):
    """Exception raised when code execution fails in sandbox."""

    pass


class ResourceLimitError(SecurityError):
    """Exception raised when resource limits are exceeded."""

    def __init__(self, message: str, resource_type: str = None):
        super().__init__(message)
        self.resource_type = resource_type


class TimeoutError(ResourceLimitError):
    """Exception raised when execution timeout is exceeded."""

    def __init__(self, message: str, timeout_seconds: float = None):
        super().__init__(message, resource_type="timeout")
        self.timeout_seconds = timeout_seconds


class MemoryLimitError(ResourceLimitError):
    """Exception raised when memory limit is exceeded."""

    def __init__(self, message: str, limit_mb: int = None):
        super().__init__(message, resource_type="memory")
        self.limit_mb = limit_mb
