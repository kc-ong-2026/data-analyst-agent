"""Common models shared across multiple agents."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChartType(str, Enum):
    """Supported chart types for visualizations."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    TABLE = "table"


class YearRange(BaseModel):
    """Represents a range of years."""

    min: int = Field(..., ge=1900, le=2100, description="Minimum year in range")
    max: int = Field(..., ge=1900, le=2100, description="Maximum year in range")

    @field_validator("max")
    @classmethod
    def max_must_be_greater_than_or_equal_to_min(cls, v: int, info) -> int:
        """Validate that max year is >= min year."""
        if "min" in info.data and v < info.data["min"]:
            raise ValueError("max year must be >= min year")
        return v

    def __str__(self) -> str:
        """String representation of year range."""
        if self.min == self.max:
            return str(self.min)
        return f"{self.min}-{self.max}"


class ColumnInfo(BaseModel):
    """Information about a dataset column."""

    name: str = Field(..., description="Column name")
    dtype: str = Field(..., description="Data type of the column")


class AgentTraceEntry(BaseModel):
    """Trace entry for agent execution tracking."""

    agent: str = Field(..., description="Name of the agent")
    success: bool = Field(..., description="Whether the agent executed successfully")


class OrchestrationMetadata(BaseModel):
    """Metadata about the orchestration workflow execution."""

    iterations: int = Field(..., ge=0, description="Number of orchestration iterations")
    workflow_plan: list[dict[str, Any]] = Field(
        default_factory=list, description="Workflow plan steps"
    )
    agents_used: list[str] = Field(default_factory=list, description="List of agents invoked")
    validation_failed: bool = Field(default=False, description="Whether validation failed")
    validation_details: dict[str, Any] | None = Field(
        default=None, description="Details about validation failure"
    )
