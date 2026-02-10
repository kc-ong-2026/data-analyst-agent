"""Models for the Query Verification Agent."""

from typing import List, Optional

from pydantic import BaseModel, Field

from .agent_common import YearRange


class DimensionCheckResult(BaseModel):
    """Result of dimension checking for employment queries."""

    checked: bool = Field(
        ..., description="Whether dimension checking was performed"
    )
    dimensions_found: List[str] = Field(
        default_factory=list,
        description="List of dimensions found in query (age, sex, industry, qualification)",
    )
    dimension_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested dimensions user could specify",
    )
    dimensions_valid: bool = Field(
        default=True, description="Whether the dimensions specification is valid"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for dimension validation failure (if dimensions_valid=False)",
    )


class QueryValidationResult(BaseModel):
    """Complete validation result for a user query."""

    valid: bool = Field(..., description="Overall validity of the query")
    topic_valid: bool = Field(..., description="Whether the topic is valid")
    reason: str = Field(..., description="Validation message or error reason")
    years_found: bool = Field(
        ..., description="Whether year/year range was found in query"
    )
    requested_years: Optional[YearRange] = Field(
        default=None, description="Requested year range extracted from query"
    )
    missing_year: bool = Field(
        default=False,
        description="Whether query is missing year specification (triggers pause)",
    )
    dimension_check: DimensionCheckResult = Field(
        default_factory=DimensionCheckResult,
        description="Result of dimension checking for employment queries",
    )
    years_available: bool = Field(
        default=True, description="Whether requested years are available in datasets"
    )
    matching_categories: List[str] = Field(
        default_factory=list,
        description="Categories that have data for requested years",
    )
    checked_availability: bool = Field(
        default=False, description="Whether year availability was checked"
    )
    year_range: Optional[YearRange] = Field(
        default=None, description="Deprecated - use requested_years instead"
    )

    @property
    def needs_user_input(self) -> bool:
        """Check if validation requires user input (missing year or unavailable years)."""
        return self.missing_year or not self.years_available

    @property
    def should_continue(self) -> bool:
        """Check if the workflow should continue to next agent."""
        return self.valid and not self.needs_user_input
