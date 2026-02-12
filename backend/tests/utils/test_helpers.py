"""
Test Helpers and Utilities.

This module provides common utility functions and helpers used across
test files including data generation, assertions, mocking, and test data
management.
"""

import logging
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pandas as pd
from faker import Faker

logger = logging.getLogger(__name__)
fake = Faker()


# ============================================================================
# Data Generation Helpers
# ============================================================================


def generate_income_data(
    num_rows: int = 100,
    years: list[int] | None = None,
    age_groups: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic income dataset.

    Args:
        num_rows: Number of rows to generate
        years: List of years (default: [2019, 2020, 2021])
        age_groups: List of age groups (default: standard age groups)

    Returns:
        DataFrame with income data
    """
    years = years or [2019, 2020, 2021]
    age_groups = age_groups or ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    sexes = ["Male", "Female"]

    data = []
    for _ in range(num_rows):
        data.append(
            {
                "year": random.choice(years),
                "age_group": random.choice(age_groups),
                "sex": random.choice(sexes),
                "average_income": round(random.uniform(2000, 8000), 2),
                "median_income": round(random.uniform(1800, 7500), 2),
            }
        )

    return pd.DataFrame(data)


def generate_employment_data(
    num_rows: int = 100,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic employment dataset.

    Args:
        num_rows: Number of rows to generate
        years: List of years

    Returns:
        DataFrame with employment data
    """
    years = years or [2019, 2020, 2021]
    age_groups = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]

    data = []
    for _ in range(num_rows):
        employment_rate = random.uniform(0.70, 0.95)
        data.append(
            {
                "year": random.choice(years),
                "age_group": random.choice(age_groups),
                "employment_rate": round(employment_rate, 4),
                "unemployment_rate": round(1 - employment_rate, 4),
                "labour_force": random.randint(10000, 500000),
            }
        )

    return pd.DataFrame(data)


def generate_hours_worked_data(
    num_rows: int = 100,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic hours worked dataset.

    Args:
        num_rows: Number of rows to generate
        years: List of years

    Returns:
        DataFrame with hours worked data
    """
    years = years or [2019, 2020, 2021]

    data = []
    for _ in range(num_rows):
        data.append(
            {
                "year": random.choice(years),
                "occupation": fake.job(),
                "average_weekly_hours": round(random.uniform(35, 50), 1),
                "median_weekly_hours": round(random.uniform(34, 48), 1),
            }
        )

    return pd.DataFrame(data)


def create_temp_csv_file(
    df: pd.DataFrame,
    file_path: Path,
    include_index: bool = False,
) -> Path:
    """
    Create temporary CSV file from DataFrame.

    Args:
        df: DataFrame to save
        file_path: Path to save file
        include_index: Whether to include index in CSV

    Returns:
        Path to created file
    """
    df.to_csv(file_path, index=include_index)
    logger.debug(f"Created temp CSV: {file_path}")
    return file_path


# ============================================================================
# Mock Helpers
# ============================================================================


def create_mock_rag_service(
    search_results: list[dict[str, Any]] | None = None,
) -> Mock:
    """
    Create mocked RAG service with configurable search results.

    Args:
        search_results: Optional list of search results to return

    Returns:
        Mocked RAG service
    """
    search_results = search_results or [
        {
            "dataset_name": "income_2020",
            "description": "Income data for 2020",
            "score": 0.85,
            "file_path": "/mock/path/income_2020.csv",
        }
    ]

    mock_service = Mock()
    mock_service.search_datasets = AsyncMock(return_value=search_results)
    mock_service.get_dataset_metadata = AsyncMock(
        return_value={
            "name": "income_2020",
            "description": "Mock income data",
            "columns": ["year", "age_group", "average_income"],
            "years": [2020],
            "categories": ["income"],
        }
    )
    mock_service.get_dataset_count = AsyncMock(return_value=len(search_results))

    return mock_service


def create_mock_llm(
    response_content: str = "Mocked LLM response",
) -> Mock:
    """
    Create mocked LLM with configurable response.

    Args:
        response_content: Content to return in response

    Returns:
        Mocked LLM
    """
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = response_content

    mock_llm.invoke = Mock(return_value=mock_response)
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    return mock_llm


def create_mock_agent_response(
    success: bool = True,
    data: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Create mock agent response.

    Args:
        success: Whether response indicates success
        data: Response data
        error: Error message if failed

    Returns:
        Mock agent response dict
    """
    return {
        "success": success,
        "data": data or {},
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# Assertion Helpers
# ============================================================================


def assert_dataframe_equal(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    check_dtype: bool = False,
) -> None:
    """
    Assert two DataFrames are equal.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        check_dtype: Whether to check dtypes

    Raises:
        AssertionError if DataFrames differ
    """
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    except AssertionError as e:
        logger.error(f"DataFrames differ: {e}")
        raise


def assert_dict_contains_keys(
    data: dict[str, Any],
    required_keys: list[str],
) -> None:
    """
    Assert dictionary contains required keys.

    Args:
        data: Dictionary to check
        required_keys: List of required keys

    Raises:
        AssertionError if keys missing
    """
    missing_keys = set(required_keys) - set(data.keys())
    assert not missing_keys, f"Missing keys: {missing_keys}"


def assert_score_in_range(
    score: float,
    min_score: float = 0.0,
    max_score: float = 1.0,
) -> None:
    """
    Assert score is within valid range.

    Args:
        score: Score to check
        min_score: Minimum valid score
        max_score: Maximum valid score

    Raises:
        AssertionError if score out of range
    """
    assert min_score <= score <= max_score, f"Score {score} not in range [{min_score}, {max_score}]"


def assert_contains_keywords(
    text: str,
    keywords: list[str],
    case_sensitive: bool = False,
) -> None:
    """
    Assert text contains all specified keywords.

    Args:
        text: Text to check
        keywords: List of keywords that must be present
        case_sensitive: Whether to check case-sensitively

    Raises:
        AssertionError if keywords missing
    """
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]

    missing_keywords = [k for k in keywords if k not in text]
    assert not missing_keywords, f"Missing keywords: {missing_keywords}\nText: {text}"


# ============================================================================
# Test Data Management
# ============================================================================


class TestDataManager:
    """
    Manager for test data lifecycle.

    Handles creating, loading, and cleaning up test data files.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize test data manager.

        Args:
            base_dir: Base directory for test data
        """
        self.base_dir = base_dir
        self.created_files: list[Path] = []

    def create_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        file_format: str = "csv",
    ) -> Path:
        """
        Create test dataset file.

        Args:
            name: Dataset name
            data: DataFrame to save
            file_format: File format (csv or excel)

        Returns:
            Path to created file
        """
        if file_format == "csv":
            file_path = self.base_dir / f"{name}.csv"
            data.to_csv(file_path, index=False)
        elif file_format == "excel":
            file_path = self.base_dir / f"{name}.xlsx"
            data.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        self.created_files.append(file_path)
        logger.debug(f"Created test dataset: {file_path}")

        return file_path

    def cleanup(self) -> None:
        """Remove all created test files."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Removed test file: {file_path}")

        self.created_files.clear()


# ============================================================================
# Performance Measurement
# ============================================================================


class PerformanceTimer:
    """
    Context manager for measuring execution time.

    Usage:
        with PerformanceTimer() as timer:
            # code to measure
            ...
        print(f"Elapsed: {timer.elapsed_ms}ms")
    """

    def __init__(self):
        """Initialize timer."""
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def __enter__(self):
        """Start timer."""
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        """Stop timer."""
        self.end_time = datetime.now()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return 0.0

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_ms / 1000.0


def measure_latency(
    func: Callable,
    num_iterations: int = 10,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Measure function latency statistics.

    Args:
        func: Function to measure
        num_iterations: Number of times to run function
        **kwargs: Arguments to pass to function

    Returns:
        Dict with latency statistics (mean, median, p90, p99)
    """
    latencies = []

    for _ in range(num_iterations):
        with PerformanceTimer() as timer:
            func(**kwargs)
        latencies.append(timer.elapsed_ms)

    latencies.sort()

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "median_ms": latencies[len(latencies) // 2],
        "p90_ms": latencies[int(len(latencies) * 0.9)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
    }


# ============================================================================
# Query Generation
# ============================================================================


def generate_test_queries(
    num_queries: int = 10,
    categories: list[str] | None = None,
    years: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate synthetic test queries.

    Args:
        num_queries: Number of queries to generate
        categories: List of categories (income, employment, hours)
        years: List of years to include

    Returns:
        List of query dicts with metadata
    """
    categories = categories or ["income", "employment", "hours_worked"]
    years = years or [2019, 2020, 2021]

    templates = {
        "income": [
            "What was the average income in {year}?",
            "Show me income by age group in {year}",
            "Compare income between males and females in {year}",
        ],
        "employment": [
            "What was the employment rate in {year}?",
            "Show unemployment by age group in {year}",
            "Compare employment rates between {year1} and {year2}",
        ],
        "hours_worked": [
            "How many hours did people work in {year}?",
            "Show working hours trend from {year1} to {year2}",
            "What were the average weekly hours in {year}?",
        ],
    }

    queries = []
    for i in range(num_queries):
        category = random.choice(categories)
        template = random.choice(templates[category])

        year = random.choice(years)
        year1, year2 = random.sample(years, 2)

        query_text = template.format(year=year, year1=year1, year2=year2)

        queries.append(
            {
                "id": f"test_query_{i+1}",
                "query": query_text,
                "category": category,
                "year": year,
            }
        )

    return queries
