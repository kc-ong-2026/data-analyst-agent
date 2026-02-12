"""
Mock LLM Responses for Integration Tests.

Provides realistic mock responses without actual API calls.
"""

from typing import Any


class MockLLMResponses:
    """Mock LLM responses for different agent types."""

    @staticmethod
    def verification_response(query: str) -> dict[str, Any]:
        """Mock verification agent response."""
        # Check for off-topic queries
        off_topic_keywords = ["weather", "sports", "entertainment", "news"]
        is_off_topic = any(kw in query.lower() for kw in off_topic_keywords)

        # Extract years from query
        import re

        years = re.findall(r"\b(19\d{2}|20[0-4]\d)\b", query)

        if is_off_topic:
            return {
                "is_relevant": False,
                "has_year": len(years) > 0,
                "years": years,
                "reason": "Query is not related to Singapore manpower statistics",
                "validation": {"is_relevant": False, "has_year": len(years) > 0, "years": years},
            }

        return {
            "is_relevant": True,
            "has_year": len(years) > 0,
            "years": years,
            "validation": {"is_relevant": True, "has_year": len(years) > 0, "years": years},
        }

    @staticmethod
    def coordinator_response(query: str) -> dict[str, Any]:
        """Mock coordinator agent response."""
        return {
            "plan": "Extract income data from datasets and analyze",
            "required_data": ["income"],
            "analysis_type": "aggregation",
            "datasets_needed": ["income_2020"],
            "needs_visualization": "chart" in query.lower() or "graph" in query.lower(),
        }

    @staticmethod
    def extraction_response(query: str) -> dict[str, Any]:
        """Mock extraction agent response."""
        return {
            "datasets": [
                {
                    "name": "income_2020",
                    "file_path": "/mock/path/income_2020.csv",
                    "columns": ["age_group", "average_income", "median_income"],
                    "score": 0.95,
                }
            ],
            "extracted_data": [
                {
                    "dataset_name": "income_2020",
                    "columns": ["age_group", "average_income", "median_income"],
                    "dtypes": {
                        "age_group": "object",
                        "average_income": "float64",
                        "median_income": "float64",
                    },
                    "data": [
                        {"age_group": "25-34", "average_income": 4500.0, "median_income": 4200.0},
                        {"age_group": "35-44", "average_income": 5800.0, "median_income": 5500.0},
                        {"age_group": "45-54", "average_income": 6200.0, "median_income": 5900.0},
                    ],
                    "source": "dataframe",
                    "metadata": {"year": 2020, "category": "income"},
                }
            ],
        }

    @staticmethod
    def analytics_response(query: str) -> dict[str, Any]:
        """Mock analytics agent response."""
        return {
            "answer": "Based on the data from 2020, the average income across all age groups was $5,500. The 35-44 age group had the highest average income at $5,800.",
            "analysis": {
                "overall_average": 5500.0,
                "highest_group": "35-44",
                "highest_value": 5800.0,
            },
            "code_generated": """
import pandas as pd
import numpy as np

# Calculate overall average
overall_avg = df['average_income'].mean()
print(f"Overall average income: ${overall_avg:.2f}")
""",
            "visualization": (
                {
                    "type": "bar",
                    "title": "Average Income by Age Group (2020)",
                    "x_axis": "Age Group",
                    "y_axis": "Average Income",
                    "data": [
                        {"label": "25-34", "value": 4500.0},
                        {"label": "35-44", "value": 5800.0},
                        {"label": "45-54", "value": 6200.0},
                    ],
                }
                if "chart" in query.lower() or "graph" in query.lower()
                else None
            ),
        }

    @staticmethod
    def orchestrator_response(query: str) -> dict[str, Any]:
        """Mock complete orchestrator response."""
        # Check if off-topic
        off_topic_keywords = ["weather", "sports", "entertainment", "news"]
        is_off_topic = any(kw in query.lower() for kw in off_topic_keywords)

        if is_off_topic:
            return {
                "success": False,
                "message": "I can only answer questions about Singapore manpower statistics (employment, income, hours worked, labour force). Your question appears to be about a different topic.",
                "error": "Off-topic query",
            }

        if not query or query.strip() == "":
            return {
                "success": False,
                "message": "Please provide a valid question.",
                "error": "Empty query",
            }

        return {
            "success": True,
            "answer": "Based on the 2020 data, the average income in Singapore was $5,500.",
            "data": {
                "overall_average": 5500.0,
                "by_age_group": [
                    {"age_group": "25-34", "average_income": 4500.0},
                    {"age_group": "35-44", "average_income": 5800.0},
                    {"age_group": "45-54", "average_income": 6200.0},
                ],
            },
            "visualization": (
                {
                    "type": "bar",
                    "title": "Average Income by Age Group (2020)",
                    "data": [
                        {"label": "25-34", "value": 4500.0},
                        {"label": "35-44", "value": 5800.0},
                        {"label": "45-54", "value": 6200.0},
                    ],
                }
                if "chart" in query.lower() or "graph" in query.lower()
                else None
            ),
            "metadata": {"query": query, "datasets_used": ["income_2020"], "processing_time": 0.5},
        }


def get_mock_llm_response(agent_type: str, query: str) -> dict[str, Any]:
    """Get mock LLM response based on agent type."""
    responses = {
        "verification": MockLLMResponses.verification_response,
        "coordinator": MockLLMResponses.coordinator_response,
        "extraction": MockLLMResponses.extraction_response,
        "analytics": MockLLMResponses.analytics_response,
        "orchestrator": MockLLMResponses.orchestrator_response,
    }

    handler = responses.get(agent_type)
    if handler:
        return handler(query)

    return {"error": f"Unknown agent type: {agent_type}"}
