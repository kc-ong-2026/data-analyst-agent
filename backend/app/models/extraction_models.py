"""Models for the Data Extraction Agent."""

from typing import Any, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, Field

from .agent_common import YearRange


class DatasetMetadata(BaseModel):
    """Metadata about a dataset."""

    table_name: str = Field(..., description="Name of the table")
    category: str = Field(..., description="Category of the dataset")
    description: str = Field(..., description="Description of the dataset")
    primary_dimensions: List[str] = Field(
        default_factory=list, description="Primary dimension columns"
    )
    numeric_columns: List[str] = Field(
        default_factory=list, description="Numeric column names"
    )
    categorical_columns: List[str] = Field(
        default_factory=list, description="Categorical column names"
    )
    year_range: Optional[YearRange] = Field(
        default=None, description="Year range covered by the dataset"
    )


class ExtractedDataset(BaseModel):
    """Data extracted from a single dataset."""

    path: str = Field(..., description="File path or source of the dataset")
    columns: List[str] = Field(..., description="Column names in the dataset")
    shape: Tuple[int, int] = Field(..., description="Shape of the dataset (rows, cols)")
    dtypes: Dict[str, str] = Field(
        default_factory=dict, description="Data types for each column"
    )
    data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Actual data rows (empty if using SQL results)"
    )
    metadata: DatasetMetadata = Field(..., description="Dataset metadata")
    table_name: str = Field(..., description="Table name in database")
    source: Literal["rag_metadata", "file_metadata", "sql"] = Field(
        ..., description="Source of the data"
    )


class MetadataSearchResult(BaseModel):
    """Result from RAG metadata search."""

    metadata_id: int = Field(..., description="ID of the metadata record")
    category: str = Field(..., description="Category of the dataset")
    file_name: str = Field(..., description="Original file name")
    table_name: str = Field(..., description="Table name in database")
    description: str = Field(..., description="Description of the dataset")
    columns: List[str] = Field(..., description="Column names")
    row_count: int = Field(..., description="Number of rows in the dataset")
    score: float = Field(..., description="Relevance score from search")


class TableSchema(BaseModel):
    """Schema information for a table."""

    table_name: str = Field(..., description="Table name")
    category: str = Field(..., description="Category of the dataset")
    description: str = Field(..., description="Description of the dataset")
    columns: List[str] = Field(..., description="Column names")
    primary_dimensions: List[str] = Field(
        default_factory=list, description="Primary dimension columns"
    )
    numeric_columns: List[str] = Field(
        default_factory=list, description="Numeric columns"
    )
    categorical_columns: List[str] = Field(
        default_factory=list, description="Categorical columns"
    )
    row_count: int = Field(..., description="Number of rows")
    year_range: Optional[Dict[str, int]] = Field(
        default=None, description="Year range (dict with min/max)"
    )
    sql_schema_prompt: str = Field(..., description="Formatted schema for SQL generation")


class RetrievalContext(BaseModel):
    """Context retrieved from RAG system."""

    metadata_results: List[MetadataSearchResult] = Field(
        default_factory=list, description="Metadata search results"
    )
    table_schemas: List[TableSchema] = Field(
        default_factory=list, description="Table schema information"
    )


class SQLQuery(BaseModel):
    """SQL query specification."""

    sql: str = Field(..., description="The SQL query string")
    table_name: str = Field(..., description="Target table name")
    description: str = Field(..., description="Description of what the query does")


class ExtractionResult(BaseModel):
    """Complete result from the extraction agent."""

    extracted_data: Dict[str, ExtractedDataset] = Field(
        ..., description="Dictionary mapping dataset names to extracted data"
    )
    datasets_used: List[str] = Field(
        default_factory=list, description="List of dataset names used"
    )
