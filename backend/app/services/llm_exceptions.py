"""
Custom exception classes for LLM service error handling and fallback mechanism.

This module defines a hierarchy of exceptions that enable proper error classification
and handling in the resilient LLM service:

- TransientLLMError: Temporary errors that should be retried (rate limits, timeouts)
- PermanentLLMError: Permanent errors that should trigger fallback (auth, bad request)
- TokenLimitError: Context length exceeded, needs model with larger context window
- CircuitBreakerOpenError: Circuit breaker is open, preventing calls to failing provider
"""

from typing import Optional


class LLMError(Exception):
    """
    Base exception for all LLM-related errors.

    This is the parent class for all custom LLM exceptions, allowing
    catch-all error handling when needed.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize LLM error.

        Args:
            message: Human-readable error message
            provider: LLM provider name (openai, anthropic, google)
            model: Model name that failed
            original_error: Original exception that was wrapped
        """
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.original_error = original_error

    def __str__(self) -> str:
        """Return formatted error message with context."""
        parts = [super().__str__()]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.model:
            parts.append(f"model={self.model}")
        return " | ".join(parts)


class TransientLLMError(LLMError):
    """
    Transient errors that should be retried with exponential backoff.

    These errors are temporary and often resolve themselves after a short wait:
    - HTTP 429: Rate limit exceeded
    - HTTP 503: Service unavailable
    - HTTP 504: Gateway timeout
    - HTTP 408: Request timeout
    - Connection errors (network issues)
    - Timeout errors (request took too long)

    The resilient LLM service will automatically retry these errors with
    exponential backoff before falling back to alternative models/providers.
    """

    pass


class PermanentLLMError(LLMError):
    """
    Permanent errors that should immediately trigger fallback to alternative model/provider.

    These errors indicate a problem that won't resolve with retries:
    - HTTP 400: Bad request (malformed input)
    - HTTP 401: Unauthorized (invalid API key)
    - HTTP 403: Forbidden (insufficient permissions)
    - HTTP 402: Payment required (billing issue)

    The resilient LLM service will skip retries and immediately trigger:
    1. Circuit breaker opening for the provider
    2. Fallback to alternative provider
    """

    pass


class TokenLimitError(LLMError):
    """
    Context length exceeded - needs model with larger context window.

    This error occurs when the input exceeds the model's maximum context length.
    Common error messages that trigger this:
    - "context_length_exceeded"
    - "maximum_context_length"
    - "token_limit"

    The resilient LLM service will:
    1. Skip retries (won't help)
    2. Try next model in chain with larger context window
    3. Fall back to alternative provider if all models exhausted
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        original_error: Optional[Exception] = None,
        tokens_sent: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize token limit error.

        Args:
            message: Human-readable error message
            provider: LLM provider name
            model: Model name that failed
            original_error: Original exception
            tokens_sent: Number of tokens in request (if known)
            max_tokens: Model's maximum token limit (if known)
        """
        super().__init__(message, provider, model, original_error)
        self.tokens_sent = tokens_sent
        self.max_tokens = max_tokens

    def __str__(self) -> str:
        """Return formatted error message with token info."""
        parts = [super().__str__()]
        if self.tokens_sent and self.max_tokens:
            parts.append(f"tokens={self.tokens_sent}/{self.max_tokens}")
        return " | ".join(parts)


class CircuitBreakerOpenError(LLMError):
    """
    Circuit breaker is open for this provider, preventing calls to failing service.

    This error is raised when the circuit breaker is in OPEN state, indicating
    that the provider has experienced too many consecutive failures and should
    not be called until the timeout period expires.

    The circuit breaker will automatically transition to HALF_OPEN state after
    the timeout, allowing limited test requests to check if the service has recovered.

    The resilient LLM service will:
    1. Skip this provider entirely
    2. Try next provider in fallback chain
    3. Log circuit breaker state change
    """

    def __init__(
        self,
        provider: str,
        failure_count: int,
        timeout_seconds: int,
    ):
        """
        Initialize circuit breaker error.

        Args:
            provider: Provider name with open circuit breaker
            failure_count: Number of consecutive failures that opened the circuit
            timeout_seconds: Seconds until circuit breaker attempts recovery
        """
        message = (
            f"Circuit breaker OPEN for provider '{provider}' "
            f"after {failure_count} failures. "
            f"Will retry in {timeout_seconds}s."
        )
        super().__init__(message, provider=provider)
        self.failure_count = failure_count
        self.timeout_seconds = timeout_seconds


class AllProvidersFailedError(LLMError):
    """
    All LLM providers and models have been exhausted without success.

    This is the final error raised when the resilient LLM service has tried:
    1. All retry attempts for each model
    2. All models in the fallback chain for each provider
    3. All providers in the fallback chain

    This indicates a systemic issue (all providers down, network issues, etc.)
    and requires manual intervention.
    """

    def __init__(
        self,
        message: str = "All LLM providers exhausted",
        attempted_providers: Optional[list] = None,
    ):
        """
        Initialize all providers failed error.

        Args:
            message: Human-readable error message
            attempted_providers: List of (provider, model) tuples that were attempted
        """
        super().__init__(message)
        self.attempted_providers = attempted_providers or []

    def __str__(self) -> str:
        """Return formatted error message with attempted providers."""
        parts = [super().__str__()]
        if self.attempted_providers:
            attempts = ", ".join(
                f"{provider}/{model}" for provider, model in self.attempted_providers
            )
            parts.append(f"attempted=[{attempts}]")
        return " | ".join(parts)
