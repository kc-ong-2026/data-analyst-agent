"""Pydantic models for the application.

This package contains all data models used across the application:
- api_models: External API request/response models
- agent_common: Shared models used across multiple agents
- verification_models: Query verification agent models
- coordinator_models: Data coordinator agent models
- extraction_models: Data extraction agent models
- analytics_models: Analytics agent models
"""

# API models (external contracts)
from .api_models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    HealthResponse,
    VisualizationData,
)

# Common agent models
from .agent_common import (
    AgentTraceEntry,
    ChartType,
    ColumnInfo,
    OrchestrationMetadata,
    YearRange,
)

# Verification agent models
from .verification_models import (
    DimensionCheckResult,
    QueryValidationResult,
)

# Coordinator agent models
from .coordinator_models import (
    CoordinatorResult,
    DelegationInfo,
    WorkflowPlan,
    WorkflowStep,
)

# Extraction agent models
from .extraction_models import (
    DatasetMetadata,
    ExtractedDataset,
    ExtractionResult,
    MetadataSearchResult,
    RetrievalContext,
    SQLQuery,
    TableSchema,
)

# Analytics agent models
from .analytics_models import (
    AnalysisResult,
)

__all__ = [
    # API Models
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ConversationHistory",
    "HealthResponse",
    "VisualizationData",
    # Common Agent Models
    "AgentTraceEntry",
    "ChartType",
    "ColumnInfo",
    "OrchestrationMetadata",
    "YearRange",
    # Verification Agent Models
    "DimensionCheckResult",
    "QueryValidationResult",
    # Coordinator Agent Models
    "CoordinatorResult",
    "DelegationInfo",
    "WorkflowPlan",
    "WorkflowStep",
    # Extraction Agent Models
    "DatasetMetadata",
    "ExtractedDataset",
    "ExtractionResult",
    "MetadataSearchResult",
    "RetrievalContext",
    "SQLQuery",
    "TableSchema",
    # Analytics Agent Models
    "AnalysisResult",
]
