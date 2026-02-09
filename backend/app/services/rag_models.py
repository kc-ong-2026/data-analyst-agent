"""Pydantic models for RAG retrieval results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkResult:
    """A single chunk retrieval result."""

    chunk_id: int
    dataset_id: int
    dataset_name: str
    chunk_type: str
    group_key: Dict[str, Any]
    chunk_text: str
    row_count: int
    numeric_summary: Dict[str, Any]
    score: float = 0.0


@dataclass
class DatasetResult:
    """Dataset metadata result."""

    dataset_id: int
    dataset_name: str
    file_path: str
    category: str
    description: str
    columns: List[Dict[str, str]]
    row_count: int
    year_range: Dict[str, Any]
    summary_text: str


@dataclass
class RetrievalResult:
    """Combined retrieval result from hybrid search."""

    query: str
    chunks: List[ChunkResult] = field(default_factory=list)
    datasets: List[DatasetResult] = field(default_factory=list)
    raw_data: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
