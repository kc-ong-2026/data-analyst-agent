"""Data processor for ingesting datasets into folder-based PostgreSQL structure."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import config
from app.db.models import (
    CATEGORY_MODELS,
    DataTableRegistry,
    EmploymentDatasetMetadata,
    HoursWorkedDatasetMetadata,
    IncomeDatasetMetadata,
    LabourForceDatasetMetadata,
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes dataset files into the folder-based RAG structure."""

    # Folder names in the dataset directory
    CATEGORY_FOLDERS = {
        "employment": "employment_dataset",
        "hours_worked": "hours_worked_dataset",
        "income": "income_dataset",
        "labour_force": "labour_force_dataset",
    }

    def __init__(self, datasets_path: Optional[str] = None):
        self.datasets_path = Path(
            datasets_path
            or config.yaml_config.get("data", {}).get("datasets_path", "../dataset")
        )

    async def process_all_datasets(self, session: AsyncSession) -> Dict[str, int]:
        """Process all dataset files organized by category folders.

        Returns:
            Counts of processed items.
        """
        counts = {"metadata_entries": 0, "data_tables": 0, "total_rows": 0}

        if not self.datasets_path.exists():
            logger.warning(f"Datasets path does not exist: {self.datasets_path}")
            return counts

        # Iterate through each category folder
        for category, folder_name in self.CATEGORY_FOLDERS.items():
            folder_path = self.datasets_path / "singapore_manpower_dataset" / folder_name
            if not folder_path.exists():
                logger.warning(f"Category folder not found: {folder_path}")
                continue

            # Find all CSV/Excel files in this folder
            data_files = list(folder_path.glob("*.csv")) + list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls"))

            logger.info(f"Processing {len(data_files)} files in category '{category}'")

            for file_path in data_files:
                try:
                    result = await self._process_single_file(session, file_path, category)
                    await session.commit()  # Commit after each successful file
                    counts["metadata_entries"] += 1
                    counts["data_tables"] += 1
                    counts["total_rows"] += result["row_count"]
                    logger.info(f"Processed: {file_path.name} ({result['row_count']} rows)")
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)
                    await session.rollback()  # Rollback on error to allow next file

        return counts

    async def _process_single_file(
        self, session: AsyncSession, file_path: Path, category: str
    ) -> Dict[str, int]:
        """Process a single dataset file.

        Args:
            session: Database session
            file_path: Path to the CSV/Excel file
            category: Category name (employment, hours_worked, income, labour_force)

        Returns:
            Dictionary with processing statistics
        """
        # Load and clean dataframe
        df = self._load_dataframe(file_path)
        df = self._clean_dataframe(df)

        # Detect schema information
        schema_info = self._detect_schema(df)

        # Generate table name for data
        data_table_name = self._generate_table_name(category, file_path.stem)

        # Generate summary text
        summary_text = self._generate_summary_text(df, file_path, schema_info)

        # Get the appropriate metadata model class
        metadata_model = CATEGORY_MODELS[category]

        # Create metadata entry
        metadata = metadata_model(
            file_name=file_path.name,
            file_path=str(file_path.relative_to(self.datasets_path)),
            description=summary_text[:500],
            table_name=data_table_name,
            columns=schema_info["columns"],
            primary_dimensions=schema_info["primary_dimensions"],
            numeric_columns=schema_info["numeric_columns"],
            categorical_columns=schema_info["categorical_columns"],
            row_count=len(df),
            year_range=schema_info["year_range"],
            summary_text=summary_text,
        )
        session.add(metadata)
        await session.flush()  # Get the ID

        # Create data table with proper SQL types and indexes
        await self._create_data_table(df, data_table_name, schema_info, session)

        # Register in data_table_registry
        registry_entry = DataTableRegistry(
            category=category,
            metadata_table=f"{category}_dataset_metadata",
            metadata_id=metadata.id,
            data_table=data_table_name,
            schema_info=schema_info,
            index_info={"dimension_indexes": schema_info["primary_dimensions"]},
        )
        session.add(registry_entry)

        return {"row_count": len(df)}

    def _load_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Load a dataset file into a pandas DataFrame."""
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and clean data."""
        # Standardize column names: lowercase, replace spaces with underscores
        df.columns = [
            re.sub(r'\s+', '_', col.strip().lower()).replace('-', '_')
            for col in df.columns
        ]

        # Replace common null-like values
        null_values = ["n.a.", "na", "n/a", "-", ".."]
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].replace({v: None for v in null_values})
                # Try to coerce to numeric
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    # Only convert if more than half are valid numbers
                    if numeric.notna().sum() > len(df) * 0.5:
                        df[col] = numeric
                except (ValueError, TypeError):
                    pass

        return df

    def _detect_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect schema information for SQL generation.

        Classifies columns as:
        - Primary dimensions: year, categorical columns (for GROUP BY)
        - Numeric columns: measures (for aggregation)
        - Categorical columns: with cardinality and sample values

        Returns:
            Schema dictionary with columns, dimensions, measures, etc.
        """
        schema = {
            "columns": [],
            "primary_dimensions": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "year_range": None,
        }

        # Detect year column
        year_col = None
        for col in df.columns:
            if "year" in col:
                year_col = col
                break
            # Check if column contains year-like values
            if df[col].dtype in ("int64", "float64"):
                vals = df[col].dropna()
                if len(vals) > 0 and vals.min() >= 1990 and vals.max() <= 2030:
                    year_col = col
                    break

        # Compute year range
        if year_col:
            vals = df[year_col].dropna()
            if len(vals) > 0:
                min_val = vals.min()
                max_val = vals.max()
                schema["year_range"] = {
                    "min": int(min_val.item() if hasattr(min_val, 'item') else min_val),
                    "max": int(max_val.item() if hasattr(max_val, 'item') else max_val),
                }
                schema["primary_dimensions"].append(year_col)

        # Process each column
        for col in df.columns:
            col_dtype = str(df[col].dtype)
            nullable = bool(df[col].isna().any())  # Convert numpy bool_ to Python bool
            sample_values = df[col].dropna().unique()[:5].tolist()

            # Convert numpy types to Python native types
            sample_values = [
                val.item() if hasattr(val, 'item') else val
                for val in sample_values
            ]

            # Add to columns list
            schema["columns"].append({
                "name": col,
                "dtype": col_dtype,
                "nullable": nullable,
                "sample_values": sample_values,
            })

            # Classify column type
            if col == year_col:
                continue  # Already handled as dimension

            if df[col].dtype == object:
                # Categorical column
                nunique = int(df[col].nunique())  # Convert numpy int to Python int
                if 2 <= nunique <= 100:  # Reasonable cardinality for grouping
                    schema["primary_dimensions"].append(col)
                    unique_vals = sorted(df[col].dropna().unique().tolist())
                    # Convert numpy types to Python native types
                    unique_vals = [
                        val.item() if hasattr(val, 'item') else val
                        for val in unique_vals
                    ]
                    schema["categorical_columns"].append({
                        "column": col,
                        "cardinality": nunique,
                        "values": unique_vals[:50],  # Limit to 50 values
                    })

            elif df[col].dtype in ("int64", "float64"):
                # Numeric column (measure)
                vals = df[col].dropna()
                if len(vals) > 0:
                    # Skip if looks like a year column
                    if not (vals.min() >= 1990 and vals.max() <= 2030):
                        schema["numeric_columns"].append(col)

        return schema

    def _generate_table_name(self, category: str, file_stem: str) -> str:
        """Generate a sanitized table name.

        Format: {category}_{sanitized_filename}
        If truncation causes collision, adds hash suffix for uniqueness.

        Args:
            category: Category name (employment, hours_worked, etc.)
            file_stem: File name without extension

        Returns:
            Sanitized table name (max 63 chars for PostgreSQL)
        """
        import hashlib

        # Sanitize file stem
        sanitized = re.sub(r'[^a-z0-9_]', '_', file_stem.lower())
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')

        # Combine with category
        table_name = f"{category}_{sanitized}"

        # If name exceeds limit, truncate and add hash for uniqueness
        if len(table_name) > 63:
            # Generate short hash from full name for uniqueness
            hash_suffix = hashlib.md5(table_name.encode()).hexdigest()[:8]
            # Reserve 9 chars for _hash (underscore + 8 hex digits)
            table_name = f"{table_name[:54]}_{hash_suffix}"

        return table_name

    def _generate_summary_text(
        self, df: pd.DataFrame, file_path: Path, schema_info: Dict[str, Any]
    ) -> str:
        """Generate a natural-language summary for the dataset."""
        name = file_path.stem.replace("_", " ")
        row_count = len(df)

        parts = [f"Dataset: {name}."]
        parts.append(f"Contains {row_count:,} rows.")

        # Year range
        year_range = schema_info.get("year_range")
        if year_range:
            parts.append(f"Covers years {year_range['min']} to {year_range['max']}.")

        # Dimensions
        if schema_info["primary_dimensions"]:
            parts.append(f"Dimensions: {', '.join(schema_info['primary_dimensions'])}.")

        # Categorical columns with values
        for cat_col in schema_info["categorical_columns"][:3]:
            values = cat_col["values"][:10]
            parts.append(
                f"Dimension '{cat_col['column']}' has {cat_col['cardinality']} categories"
                f" including: {', '.join(str(v) for v in values)}."
            )

        # Numeric measures
        if schema_info["numeric_columns"]:
            parts.append(f"Numeric measures: {', '.join(schema_info['numeric_columns'][:10])}.")

        return " ".join(parts)

    async def _create_data_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema_info: Dict[str, Any],
        session: AsyncSession,
    ) -> None:
        """Create a data table with proper SQL types and indexes.

        Args:
            df: DataFrame with data
            table_name: Name for the SQL table
            schema_info: Schema information from _detect_schema
            session: Database session
        """
        # Build column definitions with proper SQL types
        col_defs = []
        col_name_mapping = {}  # original -> safe

        for col_info in schema_info["columns"]:
            orig_col = col_info["name"]
            safe_col = re.sub(r'[^a-z0-9_]', '_', orig_col.lower()).strip('_')[:63]
            if not safe_col:
                safe_col = "col"

            col_name_mapping[orig_col] = safe_col

            # Determine SQL type
            dtype = col_info["dtype"]
            if "int" in dtype:
                sql_type = "BIGINT"
            elif "float" in dtype:
                sql_type = "DOUBLE PRECISION"
            else:
                sql_type = "TEXT"

            col_defs.append(f'"{safe_col}" {sql_type}')

        # Drop existing table if any
        await session.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))

        # Create table
        create_sql = f'CREATE TABLE "{table_name}" (id SERIAL PRIMARY KEY, {", ".join(col_defs)})'
        await session.execute(text(create_sql))

        # Create indexes on dimension columns for fast filtering
        for dimension in schema_info["primary_dimensions"]:
            if dimension in col_name_mapping:
                safe_col = col_name_mapping[dimension]
                # Create unique index name by including column name at the end
                # and ensure it fits in 63 char limit
                base_name = f"idx_{table_name}"
                remaining_chars = 63 - len(base_name) - 1  # -1 for underscore
                truncated_col = safe_col[:remaining_chars]
                index_name = f"{base_name}_{truncated_col}"

                try:
                    await session.execute(
                        text(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table_name}" ("{safe_col}")')
                    )
                except Exception as e:
                    logger.warning(f"Failed to create index {index_name}: {e}")

        # Bulk insert data
        if len(df) > 0:
            safe_col_names = [col_name_mapping[col] for col in df.columns]
            placeholders = ", ".join([f":col{i}" for i in range(len(safe_col_names))])
            col_list = ", ".join([f'"{c}"' for c in safe_col_names])
            insert_sql = f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders})'

            rows = []
            for _, row in df.iterrows():
                params = {}
                for i, orig_col in enumerate(df.columns):
                    val = row[orig_col]
                    if pd.isna(val):
                        params[f"col{i}"] = None
                    else:
                        params[f"col{i}"] = val.item() if hasattr(val, 'item') else val
                rows.append(params)

            # Insert in batches of 500
            batch_size = 500
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                await session.execute(text(insert_sql), batch)

        logger.info(f"Created table '{table_name}' with {len(df)} rows")
