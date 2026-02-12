"""Pydantic models for folder-based RAG retrieval results."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetadataResult:
    """A single folder metadata retrieval result."""

    metadata_id: int
    category: str  # employment, hours_worked, income
    file_name: str
    file_path: str
    table_name: str  # Linked data table name
    description: str
    columns: list[dict[str, Any]]  # [{name, dtype, sample_values, nullable}]
    primary_dimensions: list[str]
    numeric_columns: list[str]
    categorical_columns: list[dict[str, Any]]
    row_count: int
    year_range: dict[str, Any] | None
    summary_text: str
    score: float = 0.0


@dataclass
class TableSchema:
    """Full schema information for data loading and analysis."""

    table_name: str
    category: str
    description: str
    columns: list[dict[str, Any]]
    primary_dimensions: list[str]
    numeric_columns: list[str]
    categorical_columns: list[dict[str, Any]]
    row_count: int
    year_range: dict[str, Any] | None
    sql_schema_prompt: str  # Formatted schema for LLM
    summary_text: str = ""  # Rich context about dataset content
    file_path: str | None = None  # Path to the source file for DataFrame loading
    score: float = 0.0  # Relevance score from retrieval/reranking


@dataclass
class FolderRetrievalResult:
    """Combined retrieval result from folder-based hybrid search."""

    query: str
    metadata_results: list[MetadataResult] = field(default_factory=list)
    table_schemas: list[TableSchema] = field(default_factory=list)
    total_results: int = 0

    def get_table_names(self) -> list[str]:
        """Get list of all table names from results."""
        return [schema.table_name for schema in self.table_schemas]

    def get_schemas_by_category(self) -> dict[str, list[TableSchema]]:
        """Group table schemas by category."""
        by_category: dict[str, list[TableSchema]] = {}
        for schema in self.table_schemas:
            if schema.category not in by_category:
                by_category[schema.category] = []
            by_category[schema.category].append(schema)
        return by_category
