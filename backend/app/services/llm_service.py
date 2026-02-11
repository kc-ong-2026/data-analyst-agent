"""Model-agnostic LLM service supporting multiple providers."""

import hashlib
import logging
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import config

logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing LLM interactions with model-agnostic design."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        if provider is None:
            llm_config = config.get_llm_config()
            provider = llm_config["provider"]
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm: Optional[BaseChatModel] = None

    def _get_llm(self) -> BaseChatModel:
        """Get LLM instance based on provider configuration."""
        if self._llm is not None:
            return self._llm

        llm_config = config.get_llm_config(self.provider)
        model_name = self.model or llm_config["model"]
        # Use provided parameters or fall back to config
        temperature = self.temperature if self.temperature is not None else llm_config["temperature"]
        max_tokens = self.max_tokens if self.max_tokens is not None else llm_config["max_tokens"]

        logger.info(f"Initializing LLM: provider={self.provider}, model={model_name}, temperature={temperature}, max_tokens={max_tokens}")

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
                model_kwargs={"stream_options": {"include_usage": True}},  # Enable token usage tracking
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
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a response from the LLM."""
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

        logger.info(f"Calling LLM: {self.provider}/{self.model or 'default'}, prompt_length={len(prompt)}, num_messages={len(messages)}")
        response = await self.llm.ainvoke(messages)
        logger.info(f"LLM response received: length={len(response.content) if response.content else 0}")
        return response.content


class EmbeddingService:
    """Service for managing embeddings with model-agnostic design."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        if provider is None:
            embed_config = config.get_embedding_config()
            provider = embed_config["provider"]
        self.provider = provider
        self.model = model
        self._embeddings: Optional[Embeddings] = None

        # LRU cache for embeddings (SHA256 hash -> embedding vector)
        self._embedding_cache: Dict[str, List[float]] = {}
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

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        return await self.embeddings.aembed_documents(texts)

    async def embed_query(self, query: str, use_cache: bool = True) -> List[float]:
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
                logger.debug(f"[EMBEDDING CACHE] Evicted oldest entry, cache size={len(self._embedding_cache)}")

            # Add to cache
            self._embedding_cache[query_hash] = result

        return result

    def get_cache_stats(self) -> Dict[str, any]:
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
        logger.info(f"[EMBEDDING CACHE] Cache cleared")


def get_llm_service(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> LLMService:
    """Factory function to get LLM service."""
    return LLMService(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)


def get_embedding_service(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> EmbeddingService:
    """Factory function to get embedding service."""
    return EmbeddingService(provider=provider, model=model)
