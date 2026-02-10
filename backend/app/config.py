"""Configuration management for the application."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")

    # Database configuration
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # Server configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


class AppConfig:
    """Combined application configuration."""

    def __init__(self):
        self.settings = Settings()
        self.yaml_config = load_yaml_config()

    def get_llm_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM configuration for a specific provider."""
        # Get default provider from YAML, fallback to openai
        yaml_llm = self.yaml_config.get("llm", {})
        default_provider = yaml_llm.get("default_provider", "openai")
        provider = provider or default_provider

        llm_config = yaml_llm.get("providers", {}).get(provider, {})

        # Hardcoded fallback models per provider
        default_models = {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-sonnet-4-5-20250929",
            "google": "gemini-pro"
        }

        return {
            "provider": provider,
            "model": llm_config.get("default_model", default_models.get(provider, "gpt-4-turbo-preview")),
            "temperature": llm_config.get("temperature", 0.7),
            "max_tokens": llm_config.get("max_tokens", 4096),
            "available_models": llm_config.get("models", []),
        }

    def get_embedding_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get embedding configuration for a specific provider."""
        # Get default provider from YAML, fallback to openai
        yaml_embeddings = self.yaml_config.get("embeddings", {})
        default_provider = yaml_embeddings.get("default_provider", "openai")
        provider = provider or default_provider

        embed_config = yaml_embeddings.get("providers", {}).get(provider, {})

        # Hardcoded fallback models per provider
        default_models = {
            "openai": "text-embedding-3-small",
            "google": "models/embedding-001"
        }

        return {
            "provider": provider,
            "model": embed_config.get("default_model", default_models.get(provider, "text-embedding-3-small")),
            "dimensions": embed_config.get("dimensions", 1536),
            "available_models": embed_config.get("models", []),
        }

    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return self.yaml_config.get("vector_store", {
            "type": "chroma",
            "persist_directory": "./data/vector_store",
            "collection_name": "govtech_data",
        })

    def get_langgraph_config(self) -> Dict[str, Any]:
        """Get LangGraph configuration."""
        return self.yaml_config.get("langgraph", {
            "max_iterations": 10,
            "recursion_limit": 25,
        })

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        db_config = self.yaml_config.get("database", {})
        return {
            "url": self.settings.database_url,
            "pool_size": db_config.get("pool_size", 5),
            "max_overflow": db_config.get("max_overflow", 10),
            "echo": db_config.get("echo", False),
        }

    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration."""
        rag_config = self.yaml_config.get("rag", {})
        return {
            "embedding_batch_size": rag_config.get("embedding_batch_size", 100),
            "vector_search_top_k": rag_config.get("vector_search_top_k", 20),
            "fulltext_search_top_k": rag_config.get("fulltext_search_top_k", 20),
            "hybrid_top_k": rag_config.get("hybrid_top_k", 10),
            "rrf_k": rag_config.get("rrf_k", 60),
            "similarity_threshold": rag_config.get("similarity_threshold", 0.8),
        }

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        key_map = {
            "openai": self.settings.openai_api_key,
            "anthropic": self.settings.anthropic_api_key,
            "google": self.settings.google_api_key,
        }
        return key_map.get(provider)

    def get_langsmith_config(self) -> Dict[str, Any]:
        """Get LangSmith tracing configuration."""
        langsmith_config = self.yaml_config.get("langsmith", {})

        # Check if tracing is enabled in YAML and env var is set
        yaml_enabled = langsmith_config.get("enabled", False)
        env_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"

        return {
            "enabled": yaml_enabled and env_enabled,
            "project_name": langsmith_config.get("project_name", "govtech-multi-agent-system"),
            "tags": langsmith_config.get("tags", []),
            "api_key_configured": bool(os.environ.get("LANGCHAIN_API_KEY")),
        }


# Global config instance
config = AppConfig()
