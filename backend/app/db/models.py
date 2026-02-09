"""SQLAlchemy ORM models for the folder-based RAG system."""

from datetime import datetime, timezone
from typing import Dict, List, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase

VECTOR_DIMENSIONS = 1536


class Base(DeclarativeBase):
    pass


class FolderMetadataBase(Base):
    """
    Abstract base class for folder-level metadata tables.
    Each category (employment, hours_worked, income, labour_force) has its own metadata table.
    """
    __abstract__ = True

    id = Column(Integer, primary_key=True)
    file_name = Column(Text, nullable=False, unique=True)
    file_path = Column(Text, nullable=False)
    description = Column(Text)
    table_name = Column(Text, nullable=False, unique=True)  # Links to data table

    # Schema information for SQL generation
    columns = Column(JSONB, nullable=False)  # [{name, dtype, sample_values, nullable}]
    primary_dimensions = Column(JSONB)  # [year, sex, age, etc.]
    numeric_columns = Column(JSONB)  # [employment_rate, income, etc.]
    categorical_columns = Column(JSONB)  # [{column, cardinality, values}]

    row_count = Column(Integer, default=0)
    year_range = Column(JSONB)  # {min, max}

    # Search fields
    summary_text = Column(Text, nullable=False)
    embedding = Column(Vector(VECTOR_DIMENSIONS))
    tsv = Column(TSVECTOR)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def get_sql_schema_prompt(self) -> str:
        """
        Generate a formatted schema description for LLM SQL generation.

        Returns:
            Markdown-formatted schema suitable for inclusion in LLM prompts
        """
        schema_parts = [
            f"## Table: {self.table_name}",
            f"Description: {self.description or 'No description available'}",
            f"Row count: {self.row_count:,}",
            ""
        ]

        # Add column information
        if self.columns:
            schema_parts.append("### Columns:")
            for col in self.columns:
                col_name = col.get('name', 'unknown')
                col_type = col.get('dtype', 'unknown')
                nullable = " (nullable)" if col.get('nullable', True) else " (NOT NULL)"

                # Add sample values if available
                sample_values = col.get('sample_values', [])
                if sample_values:
                    sample_str = f" - Example values: {', '.join(map(str, sample_values[:5]))}"
                else:
                    sample_str = ""

                schema_parts.append(f"- `{col_name}` ({col_type}){nullable}{sample_str}")
            schema_parts.append("")

        # Add dimension information
        if self.primary_dimensions:
            schema_parts.append(f"### Primary Dimensions: {', '.join(self.primary_dimensions)}")
            schema_parts.append("")

        # Add numeric columns
        if self.numeric_columns:
            schema_parts.append(f"### Numeric Columns: {', '.join(self.numeric_columns)}")
            schema_parts.append("")

        # Add categorical information
        if self.categorical_columns:
            schema_parts.append("### Categorical Columns:")
            for cat_col in self.categorical_columns:
                col_name = cat_col.get('column', 'unknown')
                cardinality = cat_col.get('cardinality', 0)
                values = cat_col.get('values', [])
                if values:
                    values_str = f" - Values: {', '.join(map(str, values[:10]))}"
                    if len(values) > 10:
                        values_str += f" (+ {len(values) - 10} more)"
                else:
                    values_str = ""
                schema_parts.append(f"- `{col_name}` (cardinality: {cardinality}){values_str}")
            schema_parts.append("")

        # Add year range if available
        if self.year_range:
            min_year = self.year_range.get('min')
            max_year = self.year_range.get('max')
            if min_year and max_year:
                schema_parts.append(f"### Year Range: {min_year} - {max_year}")
                schema_parts.append("")

        return "\n".join(schema_parts)


class EmploymentDatasetMetadata(FolderMetadataBase):
    """Metadata for employment-related datasets."""
    __tablename__ = "employment_dataset_metadata"


class HoursWorkedDatasetMetadata(FolderMetadataBase):
    """Metadata for hours worked datasets."""
    __tablename__ = "hours_worked_dataset_metadata"


class IncomeDatasetMetadata(FolderMetadataBase):
    """Metadata for income-related datasets."""
    __tablename__ = "income_dataset_metadata"


class LabourForceDatasetMetadata(FolderMetadataBase):
    """Metadata for labour force datasets."""
    __tablename__ = "labour_force_dataset_metadata"


class DataTableRegistry(Base):
    """
    Registry of all dynamically created data tables.
    Tracks mapping between metadata and actual data tables for SQL generation.
    """
    __tablename__ = "data_table_registry"

    id = Column(Integer, primary_key=True)
    category = Column(Text, nullable=False)  # employment | hours_worked | income | labour_force
    metadata_table = Column(Text, nullable=False)  # e.g., 'employment_dataset_metadata'
    metadata_id = Column(Integer, nullable=False)
    data_table = Column(Text, nullable=False, unique=True)  # e.g., 'employment_resident_employment_rate_by_age_and_sex'
    schema_info = Column(JSONB, nullable=False)  # Full schema for SQL generation
    index_info = Column(JSONB)  # Created indexes
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# Mapping of category names to ORM model classes
CATEGORY_MODELS = {
    "employment": EmploymentDatasetMetadata,
    "hours_worked": HoursWorkedDatasetMetadata,
    "income": IncomeDatasetMetadata,
    "labour_force": LabourForceDatasetMetadata,
}
