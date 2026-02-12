"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime | None = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str | None = Field(None, description="User's message (optional when resuming)")
    conversation_id: str | None = Field(
        default=None, description="ID of the conversation for context"
    )
    include_visualization: bool = Field(
        default=True, description="Whether to include visualization data in response"
    )
    checkpoint_id: str | None = Field(
        default=None, description="Checkpoint ID for resuming from pause"
    )
    user_input: dict[str, Any] | None = Field(
        default=None, description="User-provided data to resume workflow"
    )


class VisualizationData(BaseModel):
    """Data for frontend visualization."""

    chart_type: str = Field(..., description="Type of chart (bar, line, pie, scatter, table)")
    title: str = Field(..., description="Title of the visualization")
    data: list[dict[str, Any]] = Field(default=[], description="Data points for the chart")
    x_axis: str | None = Field(default=None, description="X-axis data field name")
    y_axis: str | None = Field(default=None, description="Y-axis data field name")
    x_label: str | None = Field(default=None, description="Human-readable X-axis label")
    y_label: str | None = Field(default=None, description="Human-readable Y-axis label")
    description: str | None = Field(default=None, description="Description of the visualization")
    html_chart: str | None = Field(
        default=None, description="HTML representation of the chart (Plotly)"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    message: str = Field(..., description="Assistant's response message")
    conversation_id: str = Field(..., description="ID of the conversation")
    visualization: VisualizationData | None = Field(
        default=None, description="Visualization data if applicable"
    )
    sources: list[str] | None = Field(
        default=None, description="Source references used in the response"
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")


class ConversationHistory(BaseModel):
    """Conversation history model."""

    conversation_id: str
    messages: list[ChatMessage]
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    timestamp: datetime
