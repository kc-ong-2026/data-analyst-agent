"""Data API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.data_service import data_service

router = APIRouter(prefix="/data", tags=["data"])


class SearchRequest(BaseModel):
    """Request model for RAG search."""

    query: str
    category: str | None = None
    year_start: int | None = None
    year_end: int | None = None
    top_k: int = 10


@router.post("/search")
async def search_data(request: SearchRequest) -> dict[str, Any]:
    """Search datasets using hybrid RAG retrieval."""
    try:
        from app.db.session import async_session_factory

        if async_session_factory is None:
            raise HTTPException(
                status_code=503,
                detail="Database not available. RAG search requires PostgreSQL.",
            )

        from app.services.rag_service import RAGService

        rag_service = RAGService()

        year_filter = None
        if request.year_start or request.year_end:
            year_filter = {}
            if request.year_start:
                year_filter["start"] = request.year_start
            if request.year_end:
                year_filter["end"] = request.year_end

        result = await rag_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category,
            year_filter=year_filter,
        )

        return {
            "query": result.query,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "dataset_name": c.dataset_name,
                    "chunk_type": c.chunk_type,
                    "group_key": c.group_key,
                    "chunk_text": c.chunk_text[:500],
                    "score": c.score,
                }
                for c in result.chunks
            ],
            "datasets": [
                {
                    "dataset_name": d.dataset_name,
                    "category": d.category,
                    "row_count": d.row_count,
                    "year_range": d.year_range,
                    "description": d.description[:300],
                }
                for d in result.datasets
            ],
            "raw_data_keys": list(result.raw_data.keys()),
            "total_raw_rows": sum(len(v) for v in result.raw_data.values()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def list_datasets() -> dict[str, Any]:
    """List all available datasets."""
    datasets = data_service.get_available_datasets()
    return {
        "datasets": datasets,
        "count": len(datasets),
    }


@router.get("/datasets/{dataset_path:path}/info")
async def get_dataset_info(dataset_path: str) -> dict[str, Any]:
    """Get information about a specific dataset."""
    try:
        info = data_service.get_dataset_info(dataset_path)
        return {
            "dataset": dataset_path,
            **info,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_path:path}/query")
async def query_dataset(
    dataset_path: str,
    query: str | None = Query(None, description="Pandas query string"),
    columns: str | None = Query(None, description="Comma-separated column names"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum rows to return"),
) -> dict[str, Any]:
    """Query a dataset with optional filtering."""
    try:
        column_list = columns.split(",") if columns else None
        data = data_service.query_dataset(
            dataset_path=dataset_path,
            query=query,
            columns=column_list,
            limit=limit,
        )
        return {
            "dataset": dataset_path,
            "data": data,
            "count": len(data),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_path:path}/visualization")
async def get_visualization_data(
    dataset_path: str,
    x_column: str = Query(..., description="Column for X-axis"),
    y_column: str = Query(..., description="Column for Y-axis"),
    chart_type: str = Query("bar", description="Chart type: bar, line, pie, scatter"),
    limit: int = Query(20, ge=1, le=100, description="Maximum data points"),
) -> dict[str, Any]:
    """Get data formatted for visualization."""
    try:
        viz_data = data_service.get_visualization_data(
            dataset_path=dataset_path,
            x_column=x_column,
            y_column=y_column,
            chart_type=chart_type,
            limit=limit,
        )
        return viz_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
