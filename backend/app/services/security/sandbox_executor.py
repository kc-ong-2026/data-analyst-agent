"""Process-isolated code execution with resource limits."""

import logging
import multiprocessing
import traceback
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""

    success: bool
    result: Any = None
    error: str | None = None
    error_type: str | None = None
    execution_time: float | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def _apply_resource_limits(cpu_limit: int, memory_limit_mb: int, max_open_files: int):
    """Apply OS-level resource limits (Linux/macOS only)."""
    try:
        import resource

        # CPU time limit (seconds)
        if cpu_limit > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

        # Virtual memory limit (bytes)
        if memory_limit_mb > 0:
            memory_bytes = memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        # File descriptor limit (0 = block all file operations)
        if max_open_files >= 0:
            # Note: Can't set to 0 on all systems, minimum is usually 3 (stdin/stdout/stderr)
            # We set to 3 to block additional file opens
            min_fds = max(3, max_open_files)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min_fds, min_fds))

        # Disable core dumps
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    except (ImportError, ValueError, OSError) as e:
        # resource module not available on Windows, or limits can't be set
        logger.warning(f"Could not apply resource limits: {e}")


def _safe_import(name, *args, **kwargs):
    """Safe import function that only allows whitelisted modules."""
    allowed_modules = {"pandas", "numpy", "matplotlib", "matplotlib.pyplot", "datetime", "time"}
    base_module = name.split(".")[0]

    if base_module not in allowed_modules and name not in allowed_modules:
        raise ImportError(f"Import of '{name}' is not allowed for security reasons")

    return __import__(name, *args, **kwargs)


def _execute_in_process(
    code: str,
    context: dict[str, Any],
    result_queue: multiprocessing.Queue,
    cpu_limit: int,
    memory_limit_mb: int,
    max_open_files: int,
):
    """
    Execute code in isolated process with resource limits.

    This function runs in a separate process.
    """
    import time

    start_time = time.time()

    try:
        # Apply resource limits FIRST (before any other operations)
        _apply_resource_limits(cpu_limit, memory_limit_mb, max_open_files)

        # Create restricted execution environment
        exec_globals = {
            "__builtins__": {
                # Only allow safe builtins
                "print": print,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "type": type,
                "isinstance": isinstance,
                "issubclass": issubclass,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "AttributeError": AttributeError,
                "ZeroDivisionError": ZeroDivisionError,
                "__import__": _safe_import,  # Add safe import function
            }
        }

        # Add allowed imports to execution environment (pre-imported for convenience)
        exec_globals["pd"] = __import__("pandas")
        exec_globals["np"] = __import__("numpy")

        # Add context (e.g., DataFrame)
        exec_globals.update(context)

        # Execute code
        exec_locals = {}
        exec(code, exec_globals, exec_locals)

        # Get result (look for 'result' variable or last expression)
        result = exec_locals.get("result")

        execution_time = time.time() - start_time

        result_queue.put(
            ExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
            )
        )

    except MemoryError as e:
        execution_time = time.time() - start_time
        result_queue.put(
            ExecutionResult(
                success=False,
                error=f"Memory limit exceeded: {str(e)}",
                error_type="MemoryError",
                execution_time=execution_time,
            )
        )

    except Exception as e:
        execution_time = time.time() - start_time
        error_traceback = traceback.format_exc()
        result_queue.put(
            ExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                execution_time=execution_time,
                metadata={"traceback": error_traceback},
            )
        )


class SandboxExecutor:
    """Executes code in isolated process with resource limits."""

    def __init__(self, config: dict = None):
        """
        Initialize sandbox executor.

        Args:
            config: Configuration dict with isolation settings
        """
        self.config = config or {}
        self.use_process_isolation = self.config.get("use_process_isolation", True)
        self.cpu_time_limit = self.config.get("cpu_time_limit_seconds", 5)
        self.memory_limit_mb = self.config.get("memory_limit_mb", 512)
        self.wall_time_limit = self.config.get("wall_time_limit_seconds", 10)
        self.max_open_files = self.config.get("max_open_files", 3)

    def execute_code(self, code: str, context: dict[str, Any] = None) -> ExecutionResult:
        """
        Execute code in isolated sandbox.

        Args:
            code: Python code to execute
            context: Execution context (e.g., {'df': dataframe})

        Returns:
            ExecutionResult with execution status and result/error
        """
        if context is None:
            context = {}

        if not self.use_process_isolation:
            # Fallback to in-process execution (less secure, for testing)
            return self._execute_in_process_fallback(code, context)

        # Create result queue for inter-process communication
        result_queue = multiprocessing.Queue()

        # Spawn isolated process
        process = multiprocessing.Process(
            target=_execute_in_process,
            args=(
                code,
                context,
                result_queue,
                self.cpu_time_limit,
                self.memory_limit_mb,
                self.max_open_files,
            ),
        )

        try:
            process.start()
            process.join(timeout=self.wall_time_limit)

            if process.is_alive():
                # Process exceeded wall time limit
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()  # Force kill if terminate doesn't work

                return ExecutionResult(
                    success=False,
                    error=f"Execution exceeded time limit of {self.wall_time_limit} seconds",
                    error_type="TimeoutError",
                )

            # Get result from queue
            if not result_queue.empty():
                return result_queue.get_nowait()
            else:
                # Process ended but no result (crash?)
                return ExecutionResult(
                    success=False,
                    error="Process terminated unexpectedly with no result",
                    error_type="ProcessError",
                )

        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=f"Sandbox execution error: {str(e)}",
                error_type=type(e).__name__,
            )
        finally:
            # Cleanup
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)

    def _execute_in_process_fallback(self, code: str, context: dict[str, Any]) -> ExecutionResult:
        """
        Fallback execution in same process (less secure, for testing/development).

        This should only be used when process isolation is disabled.
        """
        import time

        start_time = time.time()

        try:
            # Create restricted execution environment
            exec_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "sorted": sorted,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "__import__": _safe_import,  # Add safe import function
                }
            }

            # Add allowed imports (pre-imported for convenience)
            exec_globals["pd"] = __import__("pandas")
            exec_globals["np"] = __import__("numpy")

            # Add context
            exec_globals.update(context)

            # Execute
            exec_locals = {}
            exec(code, exec_globals, exec_locals)
            result = exec_locals.get("result")

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_traceback = traceback.format_exc()
            return ExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                execution_time=execution_time,
                metadata={"traceback": error_traceback},
            )
