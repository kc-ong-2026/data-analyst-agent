"""Query Verification Agent - Validates user queries before processing."""

import re
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from app.models import DimensionCheckResult, QueryValidationResult, YearRange

from ..base_agent import (
    AgentResponse,
    AgentRole,
    AgentState,
    BaseAgent,
    GraphState,
)
from .prompts import SYSTEM_PROMPT


class QueryVerificationAgent(BaseAgent):
    """Agent responsible for validating user queries.

    Validates:
    1. Query topic relevance (employment, income, hours worked)
    2. Year/year range specification
    3. Year availability in dataset metadata
    """

    # Allowed topics (labour removed as per dataset cleanup)
    ALLOWED_TOPICS = [
        "employment",
        "employed",
        "unemployment",
        "job",
        "jobs",
        "income",
        "salary",
        "wage",
        "earnings",
        "pay",
        "workforce",
        "worker",
        "workers",
        "hours",
        "working hours",
        "work hours",
        "hours worked",
    ]

    # Dimension keywords for employment data
    AGE_KEYWORDS = [
        "age",
        "aged",
        "years old",
        "youth",
        "youths",
        "senior",
        "seniors",
        "elderly",
        "young",
        "old",
        "teenager",
        "teenagers",
    ]
    SEX_KEYWORDS = [
        "male",
        "males",
        "female",
        "females",
        "men",
        "women",
        "sex",
        "gender",
        "man",
        "woman",
    ]
    INDUSTRY_KEYWORDS = [
        "industry",
        "industries",
        "sector",
        "sectors",
        "manufacturing",
        "services",
        "service",
        "finance",
        "financial",
        "healthcare",
        "education",
        "educational",
        "construction",
        "retail",
        "hospitality",
        "technology",
        "tech",
        "technological",
    ]
    QUALIFICATION_KEYWORDS = [
        "qualification",
        "qualifications",
        "education",
        "educational",
        "degree",
        "degrees",
        "diploma",
        "diplomas",
        "university",
        "college",
        "secondary",
        "primary",
        "postgraduate",
        "graduate",
        "graduates",
        "bachelor",
        "bachelors",
        "master",
        "masters",
        "phd",
        "doctorate",
        "vocational",
        "professional",
    ]

    @property
    def role(self) -> AgentRole:
        return AgentRole.VERIFICATION

    @property
    def name(self) -> str:
        return "Query Verification"

    @property
    def description(self) -> str:
        return "Validates user queries for topic relevance and year specification"

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for query verification.

        Flow:
        validate_topic → extract_years → check_dimensions → check_year_availability → format_result → END
        """
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("validate_topic", self._validate_topic_node)
        workflow.add_node("extract_years", self._extract_years_node)
        workflow.add_node("check_dimensions", self._check_dimensions_node)
        workflow.add_node("check_year_availability", self._check_year_availability_node)
        workflow.add_node("format_result", self._format_result_node)

        # Set entry point
        workflow.set_entry_point("validate_topic")

        # Add edges with conditional routing
        workflow.add_conditional_edges(
            "validate_topic",
            self._should_continue_after_topic,
            {
                "continue": "extract_years",
                "invalid": "format_result",
            },
        )
        workflow.add_conditional_edges(
            "extract_years",
            self._should_continue_after_years,
            {
                "continue": "check_dimensions",
                "needs_input": "format_result",  # Go to format_result to return message
                "invalid": "format_result",
            },
        )
        workflow.add_conditional_edges(
            "check_dimensions",
            self._should_continue_after_dimensions,
            {
                "continue": "check_year_availability",
                "invalid": "format_result",  # Dimension validation failed
            },
        )
        workflow.add_conditional_edges(
            "check_year_availability",
            self._should_continue_after_availability,
            {
                "valid": "format_result",
                "needs_input": "format_result",  # Go to format_result to return message
            },
        )
        workflow.add_edge("format_result", END)

        return workflow.compile()

    def _should_continue_after_topic(self, state: GraphState) -> str:
        """Check if topic validation passed."""
        validation = state.get("query_validation", {})
        return "continue" if validation.get("topic_valid", False) else "invalid"

    def _should_continue_after_years(self, state: GraphState) -> str:
        """Check if year extraction found years - pause if missing for user input."""
        validation = state.get("query_validation", {})

        if validation.get("years_found", False):
            return "continue"
        elif validation.get("missing_year", False):
            return "needs_input"  # Pause instead of reject
        else:
            return "invalid"  # Truly invalid (e.g., malformed)

    def _should_continue_after_dimensions(self, state: GraphState) -> str:
        """Check if dimension validation passed for employment queries."""
        validation = state.get("query_validation", {})
        dimension_check = validation.get("dimension_check", {})

        # If not an employment query, continue
        if not dimension_check.get("checked", False):
            return "continue"

        # For employment queries, check if dimensions are valid
        if dimension_check.get("dimensions_valid", True):
            return "continue"
        else:
            return "invalid"  # Dimension validation failed

    def _should_continue_after_availability(self, state: GraphState) -> str:
        """Check if years are available - pause if out of range."""
        validation = state.get("query_validation", {})

        if validation.get("years_available", True):
            return "valid"
        else:
            return "needs_input"  # Pause to let user provide correct year

    async def _validate_topic_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Validate if query is about allowed topics."""
        current_task = state.get("current_task", "").lower()

        # Check for allowed keywords
        topic_valid = any(keyword in current_task for keyword in self.ALLOWED_TOPICS)

        validation = {
            "topic_valid": topic_valid,
            "reason": (
                "" if topic_valid else "Query must be about employment, income, or hours worked"
            ),
        }

        return {
            "query_validation": validation,
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "topic_validation": validation,
            },
        }

    async def _extract_years_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Extract year or year range from query."""
        from sqlalchemy import text

        from app.db.session import async_session_factory, get_db

        current_task = state.get("current_task", "")
        validation = state.get("query_validation", {})

        # Extract years using regex patterns
        year_patterns = [
            r"\b(19\d{2}|20[0-4]\d)\b",  # Single year (1900-2049)
            r"\b(19\d{2}|20[0-4]\d)\s*(?:to|-|through|until)\s*(19\d{2}|20[0-4]\d)\b",  # Range
        ]

        years_found = []
        for pattern in year_patterns:
            matches = re.findall(pattern, current_task, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        years_found.extend([int(y) for y in match if y])
                    else:
                        years_found.append(int(match))

        if years_found:
            min_year = min(years_found)
            max_year = max(years_found)
            validation.update(
                {
                    "years_found": True,
                    "requested_years": {"min": min_year, "max": max_year},
                }
            )
        else:
            # Fetch available years to help user
            available_years = {}
            if async_session_factory is not None:
                categories = ["employment", "income", "hours_worked"]
                try:
                    async with get_db() as session:
                        for category in categories:
                            table_name = f"{category}_dataset_metadata"
                            query = text(
                                f"""
                                SELECT
                                    MIN((year_range->>'min')::int) as min_year,
                                    MAX((year_range->>'max')::int) as max_year
                                FROM {table_name}
                                WHERE year_range IS NOT NULL
                            """
                            )
                            result = await session.execute(query)
                            row = result.fetchone()
                            if row and row[0] is not None:
                                available_years[category] = {
                                    "min": row[0],
                                    "max": row[1],
                                }
                except Exception:
                    pass

            # Format available years message
            year_info = []
            if available_years:
                for cat, yr in available_years.items():
                    year_info.append(f"{cat.replace('_', ' ').title()}: {yr['min']}-{yr['max']}")

            reason = "Which year or year range are you interested in?"
            if year_info:
                reason += "\n\nAvailable data:\n" + "\n".join(f"• {info}" for info in year_info)

            # Mark as missing instead of failing
            validation.update(
                {
                    "years_found": False,
                    "missing_year": True,
                    "reason": reason,
                }
            )

        return {
            "query_validation": validation,
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "year_extraction": validation,
            },
        }

    async def _check_dimensions_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Check if employment dimensions are specified.

        For employment queries, check if age, sex, industry, or qualification is mentioned.
        If not, add a suggestion that the query will aggregate across all dimensions.
        """
        current_task = state.get("current_task", "").lower()
        validation = state.get("query_validation", {})

        # Only check dimensions for employment-related queries
        is_employment_query = any(
            keyword in current_task
            for keyword in [
                "employment",
                "employed",
                "unemployment",
                "job",
                "jobs",
                "workforce",
                "worker",
            ]
        )

        if not is_employment_query:
            validation["dimension_check"] = {
                "checked": False,
                "reason": "Not an employment query",
            }
            return {"query_validation": validation}

        # Check which dimensions are mentioned using word boundary matching
        dimensions_found = []
        dimension_suggestions = []

        # Helper function to check if any keyword matches as a whole word
        def has_keyword_match(keywords: list, text: str) -> bool:
            """Check if any keyword matches using word boundaries."""
            import re

            for keyword in keywords:
                # Use word boundaries to match whole words only
                pattern = r"\b" + re.escape(keyword) + r"\b"
                if re.search(pattern, text, re.IGNORECASE):
                    return True
            return False

        # Check for age
        age_mentioned = has_keyword_match(self.AGE_KEYWORDS, current_task)
        if age_mentioned:
            dimensions_found.append("age")
        else:
            dimension_suggestions.append("age group (e.g., '25-54 years', 'youth', 'seniors')")

        # Check for sex
        sex_mentioned = has_keyword_match(self.SEX_KEYWORDS, current_task)
        if sex_mentioned:
            dimensions_found.append("sex")
        else:
            dimension_suggestions.append("sex (e.g., 'male', 'female')")

        # Check for industry
        industry_mentioned = has_keyword_match(self.INDUSTRY_KEYWORDS, current_task)
        if industry_mentioned:
            dimensions_found.append("industry")
        else:
            dimension_suggestions.append("industry (e.g., 'manufacturing', 'services', 'finance')")

        # Check for qualification
        qual_mentioned = has_keyword_match(self.QUALIFICATION_KEYWORDS, current_task)
        if qual_mentioned:
            dimensions_found.append("qualification")
        else:
            dimension_suggestions.append(
                "education level (e.g., 'university', 'diploma', 'secondary')"
            )

        # Add dimension info to validation
        dimension_info = {
            "checked": True,
            "dimensions_found": dimensions_found,
            "dimension_suggestions": dimension_suggestions,
        }

        # Enforce dimension requirement - fail validation if no dimensions specified
        if not dimensions_found:
            dimension_info["dimensions_valid"] = False
            dimension_info["reason"] = (
                "Please specify at least one dimension for your employment query:\n\n"
                "• **Age group** - e.g., '25-54 years', 'youth', 'seniors'\n"
                "• **Sex/Gender** - e.g., 'male', 'female'\n"
                "• **Industry/Sector** - e.g., 'technology', 'finance', 'manufacturing'\n"
                "• **Qualification** - e.g., 'university degree', 'diploma', 'secondary education'\n\n"
                'Example: "What is the employment rate for females in the technology industry?"'
            )
        else:
            dimension_info["dimensions_valid"] = True

        validation["dimension_check"] = dimension_info

        return {
            "query_validation": validation,
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "dimension_check": dimension_info,
            },
        }

    async def _check_year_availability_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Check if requested years are available in datasets."""
        from sqlalchemy import text

        from app.db.session import async_session_factory, get_db

        validation = state.get("query_validation", {})
        requested = validation.get("requested_years", {})
        req_min = requested.get("min")
        req_max = requested.get("max")

        # If database not available, skip check (allow query to proceed)
        if async_session_factory is None:
            validation["years_available"] = True
            validation["checked_availability"] = False
            return {"query_validation": validation}

        # Query all metadata tables for year ranges
        available_years = {}
        categories = ["employment", "income", "hours_worked"]

        try:
            async with get_db() as session:
                for category in categories:
                    table_name = f"{category}_dataset_metadata"

                    # Query year ranges from metadata
                    query = text(
                        f"""
                        SELECT
                            MIN((year_range->>'min')::int) as min_year,
                            MAX((year_range->>'max')::int) as max_year
                        FROM {table_name}
                        WHERE year_range IS NOT NULL
                    """
                    )

                    result = await session.execute(query)
                    row = result.fetchone()

                    if row and row[0] is not None:
                        available_years[category] = {
                            "min": row[0],
                            "max": row[1],
                        }

            # Check if requested years fall within any category's range
            years_available = False
            matching_categories = []

            for category, year_range in available_years.items():
                cat_min = year_range["min"]
                cat_max = year_range["max"]

                # Check overlap: requested range overlaps with available range
                if req_min <= cat_max and req_max >= cat_min:
                    years_available = True
                    matching_categories.append(category)

            if years_available:
                validation["years_available"] = True
                validation["matching_categories"] = matching_categories
            else:
                # Mark as needing input (will trigger pause)
                validation["years_available"] = False
                validation["missing_year"] = True  # Trigger pause for wrong year too
                validation["reason"] = f"We don't have data for years {req_min}-{req_max}."

            validation["checked_availability"] = True

        except Exception as e:
            # On error, allow query to proceed (don't block on metadata issues)
            validation["years_available"] = True
            validation["checked_availability"] = False
            validation["check_error"] = str(e)

        return {
            "query_validation": validation,
            "available_years": available_years,
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "year_availability": validation,
            },
        }

    async def _format_result_node(self, state: GraphState) -> dict[str, Any]:
        """Node: Format validation result."""
        validation = state.get("query_validation", {})

        # Check if this is an employment query and if dimensions are valid
        dimension_check = validation.get("dimension_check", {})
        is_employment_query = dimension_check.get("checked", False)
        dimensions_valid = dimension_check.get("dimensions_valid", True)

        # Determine overall validity
        # Must have valid topic, years found, years available, AND dimensions (if employment query)
        is_valid = (
            validation.get("topic_valid", False)
            and validation.get("years_found", False)
            and validation.get("years_available", True)  # Default true if not checked
            and (
                not is_employment_query or dimensions_valid
            )  # Check dimensions for employment queries
        )

        validation["valid"] = is_valid

        if not is_valid:
            # Determine failure type for conversation management
            if not validation.get("topic_valid", False):
                # Topic validation failed - clear conversation
                validation["failure_type"] = "invalid_topic"
                validation["clear_conversation"] = True
                reason = validation.get(
                    "reason", "Query must be about employment, income, or hours worked"
                )
                message = f"❌ {reason}"
                validation["reason"] = reason
            elif validation.get("missing_year"):
                # Missing year - keep conversation for context append
                validation["failure_type"] = "missing_year"
                validation["clear_conversation"] = False
                reason = validation.get("reason", "Please specify the year")
                message = reason  # Clean message asking for year
                validation["reason"] = reason
            elif is_employment_query and not dimensions_valid:
                # Missing dimension - keep conversation for context append
                validation["failure_type"] = "missing_dimension"
                validation["clear_conversation"] = False
                reason = dimension_check.get("reason", "Please specify employment dimensions")
                message = f"❌ {reason}"
                validation["reason"] = reason
            else:
                # Other failures (unavailable years, etc.) - keep conversation
                validation["failure_type"] = "other"
                validation["clear_conversation"] = False
                reason = validation.get("reason", "Query validation failed")
                message = f"❌ {reason}"
                validation["reason"] = reason
        else:
            validation["failure_type"] = None
            validation["clear_conversation"] = False
            message = "✓ Query validated successfully"

        return {
            "query_validation": validation,
            "should_continue": is_valid,
            "messages": [AIMessage(content=message)],
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                "verification_result": validation,
            },
        }

    def _build_response(self, result: GraphState, state: AgentState) -> AgentResponse:
        """Build AgentResponse from graph execution result."""
        validation_dict = result.get("query_validation", {})
        is_valid = validation_dict.get("valid", False)

        # Create Pydantic models for type safety and validation
        dimension_check_dict = validation_dict.get("dimension_check", {})
        dimension_check = DimensionCheckResult(
            checked=dimension_check_dict.get("checked", False),
            dimensions_found=dimension_check_dict.get("dimensions_found", []),
            dimension_suggestions=dimension_check_dict.get("dimension_suggestions", []),
            dimensions_valid=dimension_check_dict.get("dimensions_valid", True),
            reason=dimension_check_dict.get("reason"),
        )

        # Extract requested years if available
        requested_years_dict = validation_dict.get("requested_years")
        requested_years = None
        if requested_years_dict:
            requested_years = YearRange(
                min=requested_years_dict.get("min", 1900),
                max=requested_years_dict.get("max", 2100),
            )

        # Create validation result model
        validation_model = QueryValidationResult(
            valid=is_valid,
            topic_valid=validation_dict.get("topic_valid", False),
            reason=validation_dict.get(
                "reason", "Query validated" if is_valid else "Invalid query"
            ),
            years_found=validation_dict.get("years_found", False),
            requested_years=requested_years,
            missing_year=validation_dict.get("missing_year", False),
            dimension_check=dimension_check,
            years_available=validation_dict.get("years_available", True),
            matching_categories=validation_dict.get("matching_categories", []),
            checked_availability=validation_dict.get("checked_availability", False),
            year_range=requested_years,  # For backward compatibility
        )

        # Store validated model as dict in state metadata (GraphState compatibility)
        state.metadata["query_validation"] = validation_model.model_dump()
        state.should_continue = is_valid

        return AgentResponse(
            success=is_valid,
            message=validation_model.reason,
            data={"validation": validation_model.model_dump()},  # Convert to dict for GraphState
            next_agent=AgentRole.COORDINATOR if is_valid else None,
            state=state,
        )
