"""Model-agnostic LLM service supporting multiple providers."""

from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import config


class LLMService:
    """Service for managing LLM interactions with model-agnostic design."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider or config.settings.default_llm_provider
        self.model = model
        self._llm: Optional[BaseChatModel] = None

    def _get_llm(self) -> BaseChatModel:
        """Get LLM instance based on provider configuration."""
        if self._llm is not None:
            return self._llm

        llm_config = config.get_llm_config(self.provider)
        model_name = self.model or llm_config["model"]
        temperature = llm_config["temperature"]
        max_tokens = llm_config["max_tokens"]

        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            api_key = config.get_api_key("openai")
            if not api_key:
                raise ValueError("OpenAI API key not configured")

            self._llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )

        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            api_key = config.get_api_key("anthropic")
            if not api_key:
                raise ValueError("Anthropic API key not configured")

            self._llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )

        elif self.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = config.get_api_key("google")
            if not api_key:
                raise ValueError("Google API key not configured")

            self._llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=api_key,
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

        response = await self.llm.ainvoke(messages)
        return response.content


class EmbeddingService:
    """Service for managing embeddings with model-agnostic design."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider or config.settings.default_embedding_provider
        self.model = model
        self._embeddings: Optional[Embeddings] = None

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

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        return await self.embeddings.aembed_query(query)


def get_llm_service(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMService:
    """Factory function to get LLM service."""
    return LLMService(provider=provider, model=model)


def get_embedding_service(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> EmbeddingService:
    """Factory function to get embedding service."""
    return EmbeddingService(provider=provider, model=model)
