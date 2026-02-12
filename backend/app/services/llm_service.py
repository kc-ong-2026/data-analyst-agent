"""Model-agnostic LLM service supporting multiple providers."""

import hashlib
import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import config

logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing LLM interactions with model-agnostic design."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        if provider is None:
            llm_config = config.get_llm_config()
            provider = llm_config["provider"]
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm: BaseChatModel | None = None

    def _get_llm(self) -> BaseChatModel:
        """Get LLM instance based on provider configuration."""
        if self._llm is not None:
            return self._llm

        llm_config = config.get_llm_config(self.provider)
        model_name = self.model or llm_config["model"]
        # Use provided parameters or fall back to config
        temperature = (
            self.temperature if self.temperature is not None else llm_config["temperature"]
        )
        max_tokens = self.max_tokens if self.max_tokens is not None else llm_config["max_tokens"]

        logger.info(
            f"Initializing LLM: provider={self.provider}, model={model_name}, temperature={temperature}, max_tokens={max_tokens}"
        )

        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            provider_api_key = config.get_api_key("openai")
            if not provider_api_key:
                raise ValueError("OpenAI API key not configured")

            self._llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=provider_api_key,
                model_kwargs={
                    "stream_options": {"include_usage": True}
                },  # Enable token usage tracking
            )

        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            provider_api_key = config.get_api_key("anthropic")
            if not provider_api_key:
                raise ValueError("Anthropic API key not configured")

            self._llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                anthropic_api_key=provider_api_key,
                # Anthropic automatically includes usage metadata, no special parameter needed
            )

        elif self.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            provider_api_key = config.get_api_key("google")
            if not provider_api_key:
                raise ValueError("Google API key not configured")

            self._llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=provider_api_key,
                # Google automatically includes usage metadata
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        return self._llm

    @property
    def llm(self) -> BaseChatModel:
        """Get the LLM instance."""
        return self._get_llm()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate a response from the LLM with error classification.

        This method wraps LLM calls and converts provider-specific errors
        into custom exceptions that the resilient LLM service can handle:
        - TransientLLMError: Rate limits, timeouts (should retry)
        - PermanentLLMError: Auth errors, bad requests (should fallback)
        - TokenLimitError: Context length exceeded (need larger context model)

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt
            chat_history: Optional chat history

        Returns:
            Generated text response

        Raises:
            TransientLLMError: For transient errors (rate limits, timeouts)
            PermanentLLMError: For permanent errors (auth, validation)
            TokenLimitError: For context length exceeded errors
        """
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=prompt))

        logger.info(
            f"Calling LLM: {self.provider}/{self.model or 'default'}, "
            f"prompt_length={len(prompt)}, num_messages={len(messages)}"
        )

        try:
            response = await self.llm.ainvoke(messages)
            logger.info(
                f"LLM response received: length={len(response.content) if response.content else 0}"
            )
            return response.content

        except Exception as e:
            # Convert provider-specific errors to custom exceptions
            from app.services.llm_exceptions import (
                PermanentLLMError,
                TokenLimitError,
                TransientLLMError,
            )

            error_str = str(e).lower()

            # Check for transient errors (should retry)
            if self._is_transient_error(e):
                logger.warning(f"Transient LLM error: {self.provider}/{self.model} - {e}")
                raise TransientLLMError(
                    str(e),
                    provider=self.provider,
                    model=self.model,
                    original_error=e,
                )

            # Check for token limit errors (need larger context model)
            if any(
                keyword in error_str
                for keyword in [
                    "context_length_exceeded",
                    "maximum_context_length",
                    "token_limit",
                    "tokens",
                    "too many tokens",
                ]
            ):
                logger.warning(f"Token limit error: {self.provider}/{self.model} - {e}")
                raise TokenLimitError(
                    str(e),
                    provider=self.provider,
                    model=self.model,
                    original_error=e,
                )

            # Check for permanent errors (should fallback to different provider)
            if self._is_permanent_error(e):
                logger.error(f"Permanent LLM error: {self.provider}/{self.model} - {e}")
                raise PermanentLLMError(
                    str(e),
                    provider=self.provider,
                    model=self.model,
                    original_error=e,
                )

            # Unknown error, log and re-raise
            logger.error(f"Unknown LLM error: {self.provider}/{self.model} - {e}")
            raise

    def _is_transient_error(self, error: Exception) -> bool:
        """
        Check if error is transient and should be retried.

        Transient errors include:
        - HTTP 429: Rate limit exceeded
        - HTTP 503: Service unavailable
        - HTTP 504: Gateway timeout
        - HTTP 408: Request timeout
        - Connection errors
        - Timeout errors

        Args:
            error: Exception to check

        Returns:
            True if error is transient, False otherwise
        """
        # Check HTTP status codes
        if hasattr(error, "status_code"):
            return error.status_code in {429, 503, 504, 408}

        # Check for timeout and connection errors
        import asyncio

        try:
            import httpx

            return isinstance(
                error,
                asyncio.TimeoutError | httpx.TimeoutException | httpx.ConnectError | httpx.NetworkError | httpx.RemoteProtocolError,
            )
        except ImportError:
            # If httpx not available, just check asyncio timeout
            return isinstance(error, asyncio.TimeoutError)

    def _is_permanent_error(self, error: Exception) -> bool:
        """
        Check if error is permanent and should trigger provider fallback.

        Permanent errors include:
        - HTTP 400: Bad request (malformed input)
        - HTTP 401: Unauthorized (invalid API key)
        - HTTP 403: Forbidden (insufficient permissions)
        - HTTP 402: Payment required (billing issue)

        Args:
            error: Exception to check

        Returns:
            True if error is permanent, False otherwise
        """
        if hasattr(error, "status_code"):
            return error.status_code in {400, 401, 403, 402}
        return False


class EmbeddingService:
    """Service for managing embeddings with model-agnostic design."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
    ):
        if provider is None:
            embed_config = config.get_embedding_config()
            provider = embed_config["provider"]
        self.provider = provider
        self.model = model
        self._embeddings: Embeddings | None = None

        # LRU cache for embeddings (SHA256 hash -> embedding vector)
        self._embedding_cache: dict[str, list[float]] = {}
        self._cache_size = 1000
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_embeddings(self) -> Embeddings:
        """Get embeddings instance based on provider configuration."""
        if self._embeddings is not None:
            return self._embeddings

        embed_config = config.get_embedding_config(self.provider)
        model_name = self.model or embed_config["model"]

        if self.provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            api_key = config.get_api_key("openai")
            if not api_key:
                raise ValueError("OpenAI API key not configured")

            self._embeddings = OpenAIEmbeddings(
                model=model_name,
                api_key=api_key,
            )

        elif self.provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            api_key = config.get_api_key("google")
            if not api_key:
                raise ValueError("Google API key not configured")

            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
            )

        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

        return self._embeddings

    @property
    def embeddings(self) -> Embeddings:
        """Get the embeddings instance."""
        return self._get_embeddings()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        return await self.embeddings.aembed_documents(texts)

    async def embed_query(self, query: str, use_cache: bool = True) -> list[float]:
        """Embed a single query with LRU caching.

        Args:
            query: The query text to embed.
            use_cache: Whether to use the cache (default: True).

        Returns:
            Embedding vector as list of floats.
        """
        # Normalize query for cache key (lowercase + strip whitespace)
        normalized_query = query.lower().strip()
        query_hash = hashlib.sha256(normalized_query.encode()).hexdigest()

        # Check cache
        if use_cache and query_hash in self._embedding_cache:
            self._cache_hits += 1
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total if total > 0 else 0
            logger.info(
                f"[EMBEDDING CACHE HIT] hit_rate={hit_rate:.1%} "
                f"({self._cache_hits}/{total}), query_length={len(query)}"
            )
            return self._embedding_cache[query_hash]

        # Cache miss - generate embedding
        self._cache_misses += 1
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        logger.info(
            f"[EMBEDDING CACHE MISS] hit_rate={hit_rate:.1%} "
            f"({self._cache_hits}/{total}), embedding query: {self.provider}/{self.model or 'default'}, query_length={len(query)}"
        )

        result = await self.embeddings.aembed_query(query)
        logger.info(f"Embedding generated: dimensions={len(result)}")

        # LRU eviction (simple FIFO when full)
        if use_cache:
            if len(self._embedding_cache) >= self._cache_size:
                # Remove oldest entry (first key in dict)
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
                logger.debug(
                    f"[EMBEDDING CACHE] Evicted oldest entry, cache size={len(self._embedding_cache)}"
                )

            # Add to cache
            self._embedding_cache[query_hash] = result

        return result

    def get_cache_stats(self) -> dict[str, any]:
        """Get embedding cache statistics.

        Returns:
            Dict with cache hit rate, size, and counts.
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hit_rate": hit_rate,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total_queries": total,
            "cache_size": len(self._embedding_cache),
            "cache_max_size": self._cache_size,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("[EMBEDDING CACHE] Cache cleared")


def get_llm_service(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMService:
    """Factory function to get LLM service."""
    return LLMService(
        provider=provider, model=model, temperature=temperature, max_tokens=max_tokens
    )


def get_embedding_service(
    provider: str | None = None,
    model: str | None = None,
) -> EmbeddingService:
    """Factory function to get embedding service."""
    return EmbeddingService(provider=provider, model=model)
