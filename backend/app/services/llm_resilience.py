"""
Resilient LLM service with automatic retry, fallback, and circuit breaker pattern.

This module provides a production-grade LLM service that handles failures gracefully:

1. **Retry Logic**: Automatically retries transient errors (rate limits, timeouts)
   with exponential backoff and jitter to avoid thundering herd problems.

2. **Model Fallback**: Falls back to cheaper/faster models within the same provider
   when a model consistently fails (e.g., Opus → Sonnet → Haiku).

3. **Provider Fallback**: Falls back to alternative providers when all models in a
   provider fail (e.g., Anthropic → OpenAI → Google).

4. **Circuit Breaker**: Prevents repeated calls to failing providers, allowing them
   time to recover. States: CLOSED → OPEN → HALF_OPEN.

5. **Comprehensive Logging**: Logs all retries, fallbacks, and circuit breaker state
   changes for observability and debugging.

Usage:
    from app.services.llm_resilience import get_resilient_llm_service

    resilient_service = get_resilient_llm_service()
    response = await resilient_service.generate_with_fallback(
        messages=[HumanMessage(content="Hello")],
        primary_provider="anthropic",
        primary_model="claude-opus-4-6",
    )
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from langchain_core.messages import BaseMessage
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    wait_exponential_jitter,
    stop_after_attempt,
    RetryError,
)

from app.services.llm_exceptions import (
    TransientLLMError,
    PermanentLLMError,
    TokenLimitError,
    CircuitBreakerOpenError,
    AllProvidersFailedError,
)
from app.services.llm_service import get_llm_service
from app.config import get_config

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """
    Circuit breaker states for tracking provider health.

    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures, reject all requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for LLM provider health tracking.

    The circuit breaker prevents cascading failures by detecting when a provider
    is consistently failing and temporarily blocking requests to that provider.

    State transitions:
    - CLOSED → OPEN: After failure_threshold consecutive failures
    - OPEN → HALF_OPEN: After timeout_seconds have elapsed
    - HALF_OPEN → CLOSED: After success_threshold consecutive successes
    - HALF_OPEN → OPEN: On any failure
    """

    def __init__(
        self,
        provider: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker for a provider.

        Args:
            provider: Provider name (openai, anthropic, google)
            failure_threshold: Consecutive failures before opening circuit
            success_threshold: Consecutive successes in half-open before closing
            timeout_seconds: Seconds to wait before attempting half-open
            half_open_max_calls: Max concurrent calls allowed in half-open state
        """
        self.provider = provider
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """
        Check if circuit breaker allows execution.

        Returns:
            True if request can proceed, False if circuit is open
        """
        # If closed, always allow
        if self.state == CircuitBreakerState.CLOSED:
            return True

        # If open, check if timeout has elapsed
        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time is None:
                # Should not happen, but allow to recover
                self._transition_to_half_open()
                return True

            elapsed = (datetime.now() - self.last_failure_time).total_seconds()
            if elapsed >= self.timeout_seconds:
                self._transition_to_half_open()
                return True
            else:
                # Still in timeout period
                return False

        # If half-open, allow limited concurrent calls
        if self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self):
        """
        Record successful call, potentially closing the circuit.

        State transitions:
        - CLOSED: Reset failure count
        - HALF_OPEN: Increment success count, close if threshold reached
        - OPEN: Should not happen (can_execute() returns False)
        """
        if self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on any success
            if self.failure_count > 0:
                logger.info(
                    f"[CIRCUIT BREAKER] provider={self.provider} - "
                    f"Success after {self.failure_count} failures, resetting"
                )
                self.failure_count = 0

        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            self.half_open_calls = max(0, self.half_open_calls - 1)

            logger.info(
                f"[CIRCUIT BREAKER] provider={self.provider} - "
                f"Success in half-open state ({self.success_count}/{self.success_threshold})"
            )

            if self.success_count >= self.success_threshold:
                self._transition_to_closed()

    def record_failure(self):
        """
        Record failed call, potentially opening the circuit.

        State transitions:
        - CLOSED: Increment failure count, open if threshold reached
        - HALF_OPEN: Immediately transition back to open
        - OPEN: Already open, no action needed
        """
        self.last_failure_time = datetime.now()

        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count += 1

            logger.warning(
                f"[CIRCUIT BREAKER] provider={self.provider} - "
                f"Failure {self.failure_count}/{self.failure_threshold}"
            )

            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()

        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls = max(0, self.half_open_calls - 1)
            logger.warning(
                f"[CIRCUIT BREAKER] provider={self.provider} - "
                f"Failure in half-open state, reopening circuit"
            )
            self._transition_to_open()

    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        self.state = CircuitBreakerState.OPEN
        self.success_count = 0
        logger.error(
            f"[CIRCUIT BREAKER] provider={self.provider} - "
            f"OPENED after {self.failure_count} failures. "
            f"Will retry in {self.timeout_seconds}s"
        )

    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.half_open_calls = 0
        logger.info(
            f"[CIRCUIT BREAKER] provider={self.provider} - "
            f"Transitioned to HALF_OPEN, testing recovery"
        )

    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(
            f"[CIRCUIT BREAKER] provider={self.provider} - "
            f"CLOSED after {self.success_threshold} successes"
        )


class ResilientLLMService:
    """
    Production-grade LLM service with automatic retry, fallback, and circuit breaker.

    This service wraps the base LLMService and adds resilience features:
    1. Automatic retry with exponential backoff for transient errors
    2. Fallback to cheaper models within same provider
    3. Fallback to alternative providers
    4. Circuit breaker to prevent calls to failing providers
    5. Comprehensive logging for observability

    The service is transparent to agents - they call generate_with_fallback()
    and get a response, without needing to handle errors manually.
    """

    def __init__(self):
        """Initialize resilient LLM service with circuit breakers."""
        self.config = get_config()
        self.fallback_config = self.config.get_fallback_config()

        # Circuit breakers per provider
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._initialize_circuit_breakers()

        # Track attempted providers/models for error reporting
        self.attempted_providers: List[Tuple[str, str]] = []

    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for all configured providers."""
        if not self.fallback_config.get("circuit_breaker", {}).get("enabled", False):
            return

        cb_config = self.fallback_config["circuit_breaker"]
        for provider in self.fallback_config.get("provider_chain", []):
            self.circuit_breakers[provider] = CircuitBreaker(
                provider=provider,
                failure_threshold=cb_config.get("failure_threshold", 5),
                success_threshold=cb_config.get("success_threshold", 2),
                timeout_seconds=cb_config.get("timeout", 60),
                half_open_max_calls=cb_config.get("half_open_max_calls", 3),
            )

    async def generate_with_fallback(
        self,
        messages: List[BaseMessage],
        primary_provider: Optional[str] = None,
        primary_model: Optional[str] = None,
    ) -> str:
        """
        Generate LLM response with automatic retry and fallback.

        This is the main entry point for agents. It handles all error scenarios
        transparently:
        1. Retries transient errors with exponential backoff
        2. Falls back to cheaper models on persistent failures
        3. Falls back to alternative providers if all models fail
        4. Respects circuit breaker state

        Args:
            messages: List of messages to send to LLM
            primary_provider: Preferred provider (openai, anthropic, google)
            primary_model: Preferred model name

        Returns:
            Generated text response from LLM

        Raises:
            AllProvidersFailedError: If all providers and models have been exhausted
        """
        self.attempted_providers = []
        start_time = time.time()

        # Get provider fallback chain
        provider_chain = self._get_provider_chain(primary_provider)

        for provider in provider_chain:
            # Check circuit breaker
            if provider in self.circuit_breakers:
                if not self.circuit_breakers[provider].can_execute():
                    cb = self.circuit_breakers[provider]
                    logger.warning(
                        f"[CIRCUIT BREAKER] provider={provider} - "
                        f"Skipping due to OPEN state "
                        f"(failures={cb.failure_count})"
                    )
                    continue

            # Get model fallback chain for this provider
            model_chain = self._get_model_chain(provider, primary_model)

            for model in model_chain:
                try:
                    # Attempt generation with retry
                    response = await self._execute_with_retry(
                        provider=provider,
                        model=model,
                        messages=messages,
                    )

                    # Success! Record and return
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    logger.info(
                        f"[LLM SUCCESS] provider={provider}, model={model}, "
                        f"latency_ms={elapsed_ms}, "
                        f"attempts={len(self.attempted_providers) + 1}"
                    )

                    # Record success for circuit breaker
                    if provider in self.circuit_breakers:
                        self.circuit_breakers[provider].record_success()

                    return response

                except TransientLLMError as e:
                    # Retry exhausted for this model, try next model
                    self.attempted_providers.append((provider, model))
                    logger.warning(
                        f"[LLM RETRY EXHAUSTED] provider={provider}, model={model} - "
                        f"Falling back to next model"
                    )
                    continue

                except TokenLimitError as e:
                    # Token limit exceeded, try model with larger context
                    self.attempted_providers.append((provider, model))
                    logger.warning(
                        f"[LLM TOKEN LIMIT] provider={provider}, model={model} - "
                        f"Falling back to model with larger context"
                    )
                    continue

                except PermanentLLMError as e:
                    # Permanent error, open circuit breaker and skip provider
                    self.attempted_providers.append((provider, model))
                    if provider in self.circuit_breakers:
                        self.circuit_breakers[provider].record_failure()

                    logger.error(
                        f"[LLM PERMANENT ERROR] provider={provider}, model={model} - "
                        f"Skipping provider: {e}"
                    )
                    break  # Skip remaining models for this provider

                except Exception as e:
                    # Unknown error, record failure and continue
                    self.attempted_providers.append((provider, model))
                    if provider in self.circuit_breakers:
                        self.circuit_breakers[provider].record_failure()

                    logger.error(
                        f"[LLM UNKNOWN ERROR] provider={provider}, model={model} - "
                        f"Error: {e}"
                    )
                    continue

        # All providers exhausted
        raise AllProvidersFailedError(
            message="All LLM providers and models exhausted without success",
            attempted_providers=self.attempted_providers,
        )

    async def _execute_with_retry(
        self,
        provider: str,
        model: str,
        messages: List[BaseMessage],
    ) -> str:
        """
        Execute LLM call with exponential backoff retry for transient errors.

        Uses tenacity library to automatically retry transient errors:
        - Rate limits (429)
        - Service unavailable (503)
        - Timeouts (504, 408)
        - Connection errors

        Args:
            provider: LLM provider name
            model: Model name
            messages: Messages to send to LLM

        Returns:
            Generated text response

        Raises:
            TransientLLMError: If all retry attempts exhausted
            PermanentLLMError: On auth/validation errors
            TokenLimitError: On context length exceeded
        """
        retry_config = self.fallback_config.get("retry", {})
        max_attempts = retry_config.get("max_attempts", 3)
        initial_delay = retry_config.get("initial_delay", 1.0)
        max_delay = retry_config.get("max_delay", 30.0)
        backoff_multiplier = retry_config.get("backoff_multiplier", 2.0)

        attempt = 0

        try:
            async for attempt_obj in AsyncRetrying(
                retry=retry_if_exception_type(TransientLLMError),
                wait=wait_exponential_jitter(
                    initial=initial_delay,
                    exp_base=backoff_multiplier,
                    max=max_delay,
                    jitter=max_delay,  # Max jitter value (tenacity applies randomly)
                ),
                stop=stop_after_attempt(max_attempts),
                reraise=True,
            ):
                with attempt_obj:
                    attempt = attempt_obj.retry_state.attempt_number

                    if attempt > 1:
                        logger.info(
                            f"[LLM RETRY] provider={provider}, model={model}, "
                            f"attempt={attempt}/{max_attempts}"
                        )

                    # Get LLM service for this provider/model
                    llm_service = get_llm_service(
                        provider=provider,
                        model=model,
                    )

                    # Execute LLM call (this may raise custom exceptions)
                    response = await llm_service.llm.ainvoke(messages)
                    return response.content

        except RetryError as e:
            # Retry exhausted
            original_error = e.last_attempt.exception()
            logger.warning(
                f"[LLM RETRY EXHAUSTED] provider={provider}, model={model} - "
                f"Failed after {attempt} attempts: {original_error}"
            )
            raise TransientLLMError(
                str(original_error),
                provider=provider,
                model=model,
                original_error=original_error,
            )

    def _get_provider_chain(self, primary_provider: Optional[str]) -> List[str]:
        """
        Get provider fallback chain, with primary provider first.

        Args:
            primary_provider: Preferred provider to try first

        Returns:
            List of provider names in fallback order
        """
        provider_chain = self.fallback_config.get("provider_chain", [
            "anthropic",
            "openai",
            "google",
        ])

        # Move primary provider to front if specified
        if primary_provider and primary_provider in provider_chain:
            chain = [primary_provider]
            chain.extend(p for p in provider_chain if p != primary_provider)
            return chain

        return provider_chain

    def _get_model_chain(
        self,
        provider: str,
        primary_model: Optional[str],
    ) -> List[str]:
        """
        Get model fallback chain for provider, with primary model first.

        The fallback order typically goes from:
        - Most capable/expensive model (Opus, GPT-4)
        - Mid-tier model (Sonnet, GPT-3.5-turbo)
        - Cheapest/fastest model (Haiku)

        Args:
            provider: Provider name
            primary_model: Preferred model to try first

        Returns:
            List of model names in fallback order
        """
        model_chains = self.fallback_config.get("model_chains", {})
        model_chain = model_chains.get(provider, [])

        # If no chain configured, use default model for provider
        if not model_chain:
            llm_config = self.config.yaml_config.get("llm", {})
            providers_config = llm_config.get("providers", {})
            provider_config = providers_config.get(provider, {})
            default_model = provider_config.get("default_model")
            return [default_model] if default_model else []

        # Move primary model to front if specified
        if primary_model and primary_model in model_chain:
            chain = [primary_model]
            chain.extend(m for m in model_chain if m != primary_model)
            return chain

        return model_chain


# Singleton instance
_resilient_llm_service: Optional[ResilientLLMService] = None


def get_resilient_llm_service() -> ResilientLLMService:
    """
    Get singleton instance of resilient LLM service.

    Returns:
        ResilientLLMService instance
    """
    global _resilient_llm_service
    if _resilient_llm_service is None:
        _resilient_llm_service = ResilientLLMService()
    return _resilient_llm_service
