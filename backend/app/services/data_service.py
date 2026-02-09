"""Service for handling data loading and processing."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.config import config


class DataService:
    """Service for loading and processing datasets."""

    def __init__(self):
        self.datasets_path = Path(
            config.yaml_config.get("data", {}).get("datasets_path", "../dataset")
        )
        self.api_specs_path = Path(
            config.yaml_config.get("data", {}).get("api_specs_path", "../api_spec")
        )
        self._datasets_cache: Dict[str, pd.DataFrame] = {}

    def get_available_datasets(self) -> List[Dict[str, str]]:
        """Get list of available datasets."""
        datasets = []

        if self.datasets_path.exists():
            for file_path in self.datasets_path.rglob("*"):
                if file_path.suffix.lower() in [".csv", ".xlsx", ".xls"]:
                    datasets.append({
                        "name": file_path.stem,
                        "path": str(file_path.relative_to(self.datasets_path)),
                        "type": file_path.suffix.lower(),
                    })

        return datasets

    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load a dataset from file."""
        full_path = self.datasets_path / dataset_path

        if str(full_path) in self._datasets_cache:
            return self._datasets_cache[str(full_path)]

        if not full_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        if full_path.suffix.lower() == ".csv":
            df = pd.read_csv(full_path)
        elif full_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(full_path)
        else:
            raise ValueError(f"Unsupported file format: {full_path.suffix}")

        self._datasets_cache[str(full_path)] = df
        return df

    def get_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        df = self.load_dataset(dataset_path)
        return {
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample": df.head(5).to_dict(orient="records"),
        }

    def query_dataset(
        self,
        dataset_path: str,
        query: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query a dataset with optional filtering."""
        df = self.load_dataset(dataset_path)

        if columns:
            available_cols = [c for c in columns if c in df.columns]
            df = df[available_cols]

        if query:
            try:
                df = df.query(query)
            except Exception:
                pass

        return df.head(limit).to_dict(orient="records")

    def get_visualization_data(
        self,
        dataset_path: str,
        x_column: str,
        y_column: str,
        chart_type: str = "bar",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Prepare data for visualization."""
        df = self.load_dataset(dataset_path)

        if x_column not in df.columns or y_column not in df.columns:
            available = list(df.columns)
            raise ValueError(
                f"Columns not found. Available: {available}"
            )

        viz_df = df[[x_column, y_column]].dropna().head(limit)

        return {
            "chart_type": chart_type,
            "title": f"{y_column} by {x_column}",
            "data": viz_df.to_dict(orient="records"),
            "x_axis": x_column,
            "y_axis": y_column,
        }


# Global data service instance
data_service = DataService()
