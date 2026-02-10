"""Configuration API routes."""

from datetime import datetime

from fastapi import APIRouter

from app.config import config
from app.models import HealthResponse

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    app_config = config.yaml_config.get("app", {})

    return HealthResponse(
        status="healthy",
        version=app_config.get("version", "1.0.0"),
        timestamp=datetime.now(),
    )
