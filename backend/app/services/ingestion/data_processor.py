"""Data processor for ingesting datasets into PostgreSQL."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import config
from app.db.models import DataChunk, DatasetMetadata, RawDataTable

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes dataset files into the three-tier RAG structure."""

    def __init__(self, datasets_path: Optional[str] = None):
        self.datasets_path = Path(
            datasets_path
            or config.yaml_config.get("data", {}).get("datasets_path", "../dataset")
        )

    async def process_all_datasets(self, session: AsyncSession) -> Dict[str, int]:
        """Process all dataset files in the datasets directory.

        Returns:
            Counts of processed items.
        """
        counts = {"datasets": 0, "chunks": 0, "raw_tables": 0}

        if not self.datasets_path.exists():
            logger.warning(f"Datasets path does not exist: {self.datasets_path}")
            return counts

        files = list(self.datasets_path.rglob("*"))
        data_files = [f for f in files if f.suffix.lower() in (".csv", ".xlsx", ".xls")]

        logger.info(f"Found {len(data_files)} dataset files to process")

        for file_path in data_files:
            try:
                result = await self._process_single_dataset(session, file_path)
                counts["datasets"] += 1
                counts["chunks"] += result["chunks"]
                counts["raw_tables"] += 1
                logger.info(f"Processed: {file_path.name} ({result['chunks']} chunks)")
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

        return counts

    async def _process_single_dataset(
        self, session: AsyncSession, file_path: Path
    ) -> Dict[str, int]:
        """Process a single dataset file."""
        # Load dataframe
        df = self._load_dataframe(file_path)
        df = self._clean_dataframe(df)

        # Detect dimensions
        dimensions = self._detect_dimensions(df)

        # Determine category from directory structure
        relative_path = str(file_path.relative_to(self.datasets_path))
        category = self._detect_category(relative_path)

        # Generate dataset summary
        summary_text = self._generate_dataset_summary(df, file_path, dimensions)

        # Create dataset metadata
        metadata = DatasetMetadata(
            dataset_name=file_path.stem,
            file_path=relative_path,
            category=category,
            description=summary_text[:500],
            columns=[{"name": col, "dtype": str(df[col].dtype)} for col in df.columns],
            row_count=len(df),
            year_range=dimensions.get("year_range", {}),
            dimensions=dimensions,
            summary_text=summary_text,
        )
        session.add(metadata)
        await session.flush()  # Get the ID

        # Create chunks
        chunks = self._create_chunks(df, metadata, dimensions)
        for chunk in chunks:
            session.add(chunk)

        # Create raw data table
        await self._create_raw_table(df, metadata, session)

        return {"chunks": len(chunks)}

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
                df[col] = df[col].replace(
                    {v: None for v in null_values},
                )
                # Try to coerce to numeric
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    # Only convert if more than half are valid numbers
                    if numeric.notna().sum() > len(df) * 0.5:
                        df[col] = numeric
                except (ValueError, TypeError):
                    pass

        return df

    def _detect_dimensions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify categorical dimensions and year/period columns."""
        dimensions: Dict[str, Any] = {}

        # Detect year column
        year_col = None
        for col in df.columns:
            if "year" in col:
                year_col = col
                break
            # Check if column contains year-like values
            if df[col].dtype in ("int64", "float64"):
                vals = df[col].dropna()
                if len(vals) > 0:
                    if vals.min() >= 1990 and vals.max() <= 2030:
                        year_col = col
                        break

        if year_col:
            vals = df[year_col].dropna()
            dimensions["year_column"] = year_col
            dimensions["year_range"] = {
                "min": int(vals.min()) if len(vals) > 0 else None,
                "max": int(vals.max()) if len(vals) > 0 else None,
            }

        # Detect categorical columns (potential grouping dimensions)
        categorical_cols = []
        for col in df.columns:
            if col == year_col:
                continue
            if df[col].dtype == object:
                nunique = df[col].nunique()
                # Categorical if between 2 and 50 unique values
                if 2 <= nunique <= 50:
                    categorical_cols.append({
                        "column": col,
                        "unique_values": sorted(df[col].dropna().unique().tolist()),
                        "count": nunique,
                    })

        dimensions["categorical_columns"] = categorical_cols

        # Detect numeric columns
        numeric_cols = []
        for col in df.columns:
            if col == year_col:
                continue
            if df[col].dtype in ("int64", "float64"):
                vals = df[col].dropna()
                if len(vals) > 0 and not (vals.min() >= 1990 and vals.max() <= 2030):
                    numeric_cols.append(col)

        dimensions["numeric_columns"] = numeric_cols

        return dimensions

    def _detect_category(self, relative_path: str) -> str:
        """Detect the dataset category from directory structure."""
        parts = relative_path.lower().split("/")
        for part in parts:
            if "employment" in part:
                return "employment"
            elif "income" in part:
                return "income"
            elif "hours" in part:
                return "hours_worked"
            elif "labour" in part or "labor" in part:
                return "labour_force"

        return "general"

    def _generate_dataset_summary(
        self, df: pd.DataFrame, file_path: Path, dimensions: Dict[str, Any]
    ) -> str:
        """Generate a natural-language summary for the dataset."""
        name = file_path.stem.replace("_", " ")
        cols = list(df.columns)
        row_count = len(df)

        parts = [f"Dataset: {name}."]
        parts.append(f"Contains {row_count} rows and {len(cols)} columns.")
        parts.append(f"Columns: {', '.join(cols[:15])}.")

        year_range = dimensions.get("year_range", {})
        if year_range.get("min") and year_range.get("max"):
            parts.append(
                f"Covers years {year_range['min']} to {year_range['max']}."
            )

        for cat_col in dimensions.get("categorical_columns", [])[:3]:
            values = cat_col["unique_values"][:10]
            parts.append(
                f"Dimension '{cat_col['column']}' has {cat_col['count']} categories"
                f" including: {', '.join(str(v) for v in values)}."
            )

        numeric_cols = dimensions.get("numeric_columns", [])
        if numeric_cols:
            parts.append(f"Numeric measures: {', '.join(numeric_cols[:10])}.")

        return " ".join(parts)

    def _create_chunks(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        dimensions: Dict[str, Any],
    ) -> List[DataChunk]:
        """Create group-level chunks from the dataframe."""
        chunks = []

        # Determine grouping columns
        year_col = dimensions.get("year_column")
        cat_cols = [c["column"] for c in dimensions.get("categorical_columns", [])]

        # Pick primary grouping: year + first categorical column
        group_cols = []
        if year_col:
            group_cols.append(year_col)
        if cat_cols:
            group_cols.append(cat_cols[0])

        if not group_cols:
            # No grouping possible — create a single chunk for the whole dataset
            chunk_text = self._render_chunk_text(
                df, metadata.dataset_name, {}
            )
            chunks.append(
                DataChunk(
                    dataset_id=metadata.id,
                    chunk_type="full",
                    group_key={},
                    chunk_text=chunk_text,
                    row_count=len(df),
                    numeric_summary=self._compute_numeric_summary(df, dimensions),
                )
            )
            return chunks

        # Group and create chunks
        try:
            grouped = df.groupby(group_cols, dropna=False)
        except Exception as e:
            logger.warning(f"Grouping failed for {metadata.dataset_name}: {e}")
            chunk_text = self._render_chunk_text(df, metadata.dataset_name, {})
            chunks.append(
                DataChunk(
                    dataset_id=metadata.id,
                    chunk_type="full",
                    group_key={},
                    chunk_text=chunk_text,
                    row_count=len(df),
                    numeric_summary=self._compute_numeric_summary(df, dimensions),
                )
            )
            return chunks

        for group_key_vals, group_df in grouped:
            if not isinstance(group_key_vals, tuple):
                group_key_vals = (group_key_vals,)

            group_key = {}
            for col, val in zip(group_cols, group_key_vals):
                group_key[col] = str(val) if pd.notna(val) else None

            chunk_text = self._render_chunk_text(
                group_df, metadata.dataset_name, group_key
            )

            chunks.append(
                DataChunk(
                    dataset_id=metadata.id,
                    chunk_type="group",
                    group_key=group_key,
                    chunk_text=chunk_text,
                    row_count=len(group_df),
                    numeric_summary=self._compute_numeric_summary(group_df, dimensions),
                )
            )

        return chunks

    def _render_chunk_text(
        self, df: pd.DataFrame, dataset_name: str, group_key: Dict[str, Any]
    ) -> str:
        """Render a natural-language summary for a chunk group."""
        name = dataset_name.replace("_", " ")
        parts = [f"From dataset '{name}'."]

        if group_key:
            conditions = [f"{k}={v}" for k, v in group_key.items() if v is not None]
            if conditions:
                parts.append(f"Group: {', '.join(conditions)}.")

        # Summarize numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols[:5]:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            col_name = col.replace("_", " ")
            if len(vals) == 1:
                parts.append(f"{col_name}: {vals.iloc[0]}.")
            else:
                parts.append(
                    f"{col_name}: min={vals.min()}, max={vals.max()}, "
                    f"mean={vals.mean():.2f}."
                )

        # Include first few rows as text
        sample = df.head(3)
        for _, row in sample.iterrows():
            row_parts = []
            for col in df.columns[:8]:
                val = row[col]
                if pd.notna(val):
                    row_parts.append(f"{col.replace('_', ' ')}={val}")
            if row_parts:
                parts.append(f"Row: {', '.join(row_parts)}.")

        return " ".join(parts)

    def _compute_numeric_summary(
        self, df: pd.DataFrame, dimensions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute summary statistics for numeric columns."""
        summary = {}
        numeric_cols = dimensions.get("numeric_columns", [])

        for col in numeric_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            summary[col] = {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "count": int(vals.count()),
            }

        return summary

    async def _create_raw_table(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        session: AsyncSession,
    ) -> None:
        """Create a raw data table and bulk insert rows."""
        # Sanitize table name
        table_name = "raw_" + re.sub(r'[^a-z0-9]', '_', metadata.dataset_name.lower())
        table_name = re.sub(r'_+', '_', table_name).strip('_')[:63]

        # Build column definitions
        col_defs = []
        col_names = []
        for col in df.columns:
            safe_col = re.sub(r'[^a-z0-9_]', '_', col.lower()).strip('_')[:63]
            if not safe_col:
                safe_col = "col"
            col_names.append(safe_col)

            if df[col].dtype in ("int64",):
                col_defs.append(f'"{safe_col}" BIGINT')
            elif df[col].dtype in ("float64",):
                col_defs.append(f'"{safe_col}" DOUBLE PRECISION')
            else:
                col_defs.append(f'"{safe_col}" TEXT')

        # Drop existing table if any
        await session.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))

        # Create table
        create_sql = f'CREATE TABLE "{table_name}" (id SERIAL PRIMARY KEY, {", ".join(col_defs)})'
        await session.execute(text(create_sql))

        # Bulk insert using executemany with parameterized queries
        if len(df) > 0:
            placeholders = ", ".join([f":col{i}" for i in range(len(col_names))])
            col_list = ", ".join([f'"{c}"' for c in col_names])
            insert_sql = f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders})'

            rows = []
            for _, row in df.iterrows():
                params = {}
                for i, (orig_col, safe_col) in enumerate(zip(df.columns, col_names)):
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

        # Register in raw_data_tables
        raw_table = RawDataTable(
            dataset_id=metadata.id,
            table_name=table_name,
            columns=[{"original": orig, "safe": safe} for orig, safe in zip(df.columns, col_names)],
            row_count=len(df),
        )
        session.add(raw_table)
