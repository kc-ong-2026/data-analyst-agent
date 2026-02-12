"""Audit logging for code execution security events."""

import hashlib
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class CodeExecutionAuditLogger:
    """Logs all code executions for security audit trail."""

    def __init__(self, config: dict = None):
        """
        Initialize audit logger.

        Args:
            config: Configuration dict with audit settings
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        log_file = self.config.get("log_file", "/app/logs/code_execution_audit.log")

        if not self.enabled:
            self.logger = None
            return

        # Create logger
        self.logger = logging.getLogger("code_execution_audit")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Don't propagate to root logger

        # Ensure log directory exists - with fallback for CI/test environments
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            # Fallback to local logs directory for CI/test environments
            fallback_dir = Path("logs")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(fallback_dir / "code_execution_audit.log")
            log_path = Path(log_file)

        # Add rotating file handler (100MB max, 10 backups)
        handler = RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10,
        )

        # JSON formatter for structured logging
        formatter = logging.Formatter(
            "%(message)s"  # We'll format as JSON ourselves
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_execution(
        self,
        code: str,
        query: str = None,
        validation_result: Any = None,
        execution_result: Any = None,
        user_context: dict[str, Any] = None,
    ) -> None:
        """
        Log a code execution event.

        Args:
            code: The code that was executed
            query: Original user query that generated the code
            validation_result: Result of code validation
            execution_result: Result of code execution
            user_context: Optional user context (session ID, user ID, etc.)
        """
        if not self.enabled or self.logger is None:
            return

        # Generate code hash for deduplication
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        # Build audit log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "code_execution",
            "code_hash": code_hash,
            "code": code[:1000],  # Truncate to 1000 chars
            "code_length": len(code),
        }

        if query:
            log_entry["query"] = query[:500]  # Truncate query too

        # Validation info
        if validation_result:
            log_entry["validation"] = {
                "valid": getattr(validation_result, "is_valid", None),
                "errors": getattr(validation_result, "errors", []),
                "warnings": getattr(validation_result, "warnings", []),
            }

        # Execution info
        if execution_result:
            log_entry["execution"] = {
                "success": getattr(execution_result, "success", None),
                "error_type": getattr(execution_result, "error_type", None),
                "error": getattr(execution_result, "error", None),
                "execution_time": getattr(execution_result, "execution_time", None),
            }

            # Add result type (but not the actual result data for privacy)
            if hasattr(execution_result, "result") and execution_result.result is not None:
                log_entry["execution"]["result_type"] = type(
                    execution_result.result
                ).__name__

        # User context
        if user_context:
            log_entry["user_context"] = user_context

        # Log as JSON
        self.logger.info(json.dumps(log_entry))

    def log_security_violation(
        self,
        code: str,
        violation_type: str,
        details: str,
        severity: str = "high",
    ) -> None:
        """
        Log a security violation.

        Args:
            code: The code that triggered the violation
            violation_type: Type of violation (e.g., 'forbidden_attribute')
            details: Detailed description
            severity: Severity level (low, medium, high, critical)
        """
        if not self.enabled or self.logger is None:
            return

        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "security_violation",
            "severity": severity,
            "violation_type": violation_type,
            "details": details,
            "code_hash": code_hash,
            "code": code[:500],  # Truncate for security log
        }

        self.logger.warning(json.dumps(log_entry))

    def log_resource_limit_exceeded(
        self,
        code: str,
        resource_type: str,
        limit: Any,
        actual: Any = None,
    ) -> None:
        """
        Log a resource limit exceeded event.

        Args:
            code: The code that exceeded limits
            resource_type: Type of resource (cpu, memory, timeout)
            limit: The limit that was exceeded
            actual: Actual value if known
        """
        if not self.enabled or self.logger is None:
            return

        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "resource_limit_exceeded",
            "resource_type": resource_type,
            "limit": limit,
            "code_hash": code_hash,
        }

        if actual is not None:
            log_entry["actual"] = actual

        self.logger.warning(json.dumps(log_entry))
