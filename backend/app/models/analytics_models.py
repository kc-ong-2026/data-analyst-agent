"""Models for the Analytics Agent."""

from pydantic import BaseModel, Field

from .api_models import VisualizationData


class AnalysisResult(BaseModel):
    """Result from the analytics agent."""

    text: str = Field(..., description="The analysis text response")
    visualization: VisualizationData | None = Field(
        default=None, description="Visualization specification if applicable"
    )
    data_sources: list[str] = Field(
        default_factory=list, description="List of data sources used in the analysis"
    )
