"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="User's message")
    conversation_id: Optional[str] = Field(
        default=None, description="ID of the conversation for context"
    )
    include_visualization: bool = Field(
        default=True, description="Whether to include visualization data in response"
    )


class VisualizationData(BaseModel):
    """Data for frontend visualization."""

    chart_type: str = Field(
        ..., description="Type of chart (bar, line, pie, scatter, table)"
    )
    title: str = Field(..., description="Title of the visualization")
    data: List[Dict[str, Any]] = Field(..., description="Data points for the chart")
    x_axis: Optional[str] = Field(default=None, description="X-axis label/field")
    y_axis: Optional[str] = Field(default=None, description="Y-axis label/field")
    description: Optional[str] = Field(
        default=None, description="Description of the visualization"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    message: str = Field(..., description="Assistant's response message")
    conversation_id: str = Field(..., description="ID of the conversation")
    visualization: Optional[VisualizationData] = Field(
        default=None, description="Visualization data if applicable"
    )
    sources: Optional[List[str]] = Field(
        default=None, description="Source references used in the response"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class ConversationHistory(BaseModel):
    """Conversation history model."""

    conversation_id: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    timestamp: datetime
