"""Configuration API routes."""

from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter

from app.config import config
from app.models import ConfigResponse, HealthResponse, ModelConfig

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Get current configuration."""
    llm_config = config.get_llm_config()
    embed_config = config.get_embedding_config()

    return ConfigResponse(
        llm_providers=["openai", "anthropic", "google"],
        embedding_providers=["openai", "google"],
        current_llm=ModelConfig(
            provider=llm_config["provider"],
            model=llm_config["model"],
            available_models=llm_config["available_models"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
        ),
        current_embedding={
            "provider": embed_config["provider"],
            "model": embed_config["model"],
            "dimensions": embed_config["dimensions"],
        },
    )


@router.get("/providers")
async def get_providers() -> Dict:
    """Get available LLM and embedding providers with their models."""
    yaml_config = config.yaml_config

    llm_providers = {}
    for provider, conf in yaml_config.get("llm", {}).get("providers", {}).items():
        llm_providers[provider] = {
            "models": conf.get("models", []),
            "default_model": conf.get("default_model"),
            "has_api_key": config.get_api_key(provider) is not None,
        }

    embedding_providers = {}
    for provider, conf in yaml_config.get("embeddings", {}).get("providers", {}).items():
        embedding_providers[provider] = {
            "models": conf.get("models", []),
            "default_model": conf.get("default_model"),
            "has_api_key": config.get_api_key(provider) is not None,
        }

    return {
        "llm_providers": llm_providers,
        "embedding_providers": embedding_providers,
    }


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    app_config = config.yaml_config.get("app", {})

    return HealthResponse(
        status="healthy",
        version=app_config.get("version", "1.0.0"),
        timestamp=datetime.now(),
    )
