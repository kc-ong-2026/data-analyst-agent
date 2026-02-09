"""Pydantic models for folder-based RAG retrieval results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetadataResult:
    """A single folder metadata retrieval result."""

    metadata_id: int
    category: str  # employment, hours_worked, income, labour_force
    file_name: str
    file_path: str
    table_name: str  # Linked data table name
    description: str
    columns: List[Dict[str, Any]]  # [{name, dtype, sample_values, nullable}]
    primary_dimensions: List[str]
    numeric_columns: List[str]
    categorical_columns: List[Dict[str, Any]]
    row_count: int
    year_range: Optional[Dict[str, Any]]
    summary_text: str
    score: float = 0.0


@dataclass
class TableSchema:
    """Full schema information for SQL generation."""

    table_name: str
    category: str
    description: str
    columns: List[Dict[str, Any]]
    primary_dimensions: List[str]
    numeric_columns: List[str]
    categorical_columns: List[Dict[str, Any]]
    row_count: int
    year_range: Optional[Dict[str, Any]]
    sql_schema_prompt: str  # Formatted schema for LLM


@dataclass
class FolderRetrievalResult:
    """Combined retrieval result from folder-based hybrid search."""

    query: str
    metadata_results: List[MetadataResult] = field(default_factory=list)
    table_schemas: List[TableSchema] = field(default_factory=list)
    total_results: int = 0

    def get_table_names(self) -> List[str]:
        """Get list of all table names from results."""
        return [schema.table_name for schema in self.table_schemas]

    def get_schemas_by_category(self) -> Dict[str, List[TableSchema]]:
        """Group table schemas by category."""
        by_category: Dict[str, List[TableSchema]] = {}
        for schema in self.table_schemas:
            if schema.category not in by_category:
                by_category[schema.category] = []
            by_category[schema.category].append(schema)
        return by_category
