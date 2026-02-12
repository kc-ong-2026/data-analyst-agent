# Architecture Documentation

This document provides a comprehensive overview of the Singapore Government Data Chat Assistant system architecture, including system design, agent workflows, database schema, and key design decisions.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Database Schema](#database-schema)
4. [Multi-Agent System](#multi-agent-system)
5. [RAG System](#rag-system)
6. [LLM Resilience & Fallback](#llm-resilience--fallback)
7. [Frontend Architecture](#frontend-architecture)
8. [Design Decisions](#design-decisions)
9. [Data Flow](#data-flow)

---

## System Overview

The Singapore Government Data Chat Assistant is a multi-agent AI system that enables users to query and analyze Singapore government datasets through natural language. The system uses LangGraph for agent orchestration, PostgreSQL with pgvector for RAG (Retrieval Augmented Generation), and provides intelligent fallback mechanisms for LLM failures.

### Key Features

- **Multi-Agent Workflow**: Four specialized agents (Verification, Coordinator, Extraction, Analytics) working together
- **Hybrid RAG**: Vector search + BM25 search with RRF fusion and cross-encoder reranking
- **LLM Resilience**: Automatic retry, model fallback, and provider fallback with circuit breaker pattern
- **Code Generation**: Pandas code generation for safe data analysis (no direct SQL execution)
- **Confidence-Based Loading**: Smart dataset selection based on relevance scores
- **Real-time Streaming**: Server-sent events for progressive response delivery

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React + TypeScript)            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Chat         │  │ Visualization│  │ Zustand      │          │
│  │ Interface    │  │ Panel        │  │ Store        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP/SSE
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI + Python)                    │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Agent Orchestrator (LangGraph)                 │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │ │
│  │  │Verifica- │→ │Coordina- │→ │Extrac-   │→ │Analytics │  │ │
│  │  │tion      │  │tor       │  │tion      │  │          │  │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ RAG Service     │  │ LLM Resilience  │  │ Data Service    │ │
│  │ (Vector+BM25+   │  │ (Retry+Fallback)│  │ (DataFrame Ops) │ │
│  │  Reranking)     │  └─────────────────┘  └─────────────────┘ │
│  └─────────────────┘                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            PostgreSQL with pgvector Extension                    │
│  ┌──────────────────────────┐  ┌──────────────────────────┐    │
│  │ employment_dataset_      │  │ income_dataset_          │    │
│  │ metadata (folder-based)  │  │ metadata (folder-based)  │    │
│  └──────────────────────────┘  └──────────────────────────┘    │
│  ┌──────────────────────────┐                                   │
│  │ hours_worked_dataset_    │                                   │
│  │ metadata (folder-based)  │                                   │
│  └──────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

The system uses a three-tier metadata architecture designed for folder-based dataset management.

### Metadata Tables (Folder-Based)

Three category-specific metadata tables store folder-level information:

**1. employment_dataset_metadata**
```sql
CREATE TABLE employment_dataset_metadata (
    id SERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,              -- Dataset file name
    file_path TEXT NOT NULL,              -- Absolute path to file
    table_name TEXT UNIQUE,               -- Sanitized table name
    description TEXT,                     -- Dataset description
    columns JSONB,                        -- Column metadata [{name, dtype}]
    primary_dimensions JSONB,             -- Key dimensions (e.g., ['age', 'sex'])
    numeric_columns JSONB,                -- Numeric column names
    categorical_columns JSONB,            -- Categorical column names
    row_count INTEGER,                    -- Total rows in dataset
    year_range JSONB,                     -- {min: 2010, max: 2023}
    summary_text TEXT,                    -- Rich summary for RAG
    embedding vector(1536),               -- OpenAI text-embedding-3-small
    tsv tsvector,                         -- Full-text search vector
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_employment_metadata_embedding ON employment_dataset_metadata
    USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_employment_metadata_tsv ON employment_dataset_metadata USING GIN (tsv);
```

**2. income_dataset_metadata** (Same structure as employment)

**3. hours_worked_dataset_metadata** (Same structure as employment)

### Why Folder-Based Metadata?

- **Scalability**: Each category has its own table, avoiding single table bottlenecks
- **Domain-Specific Indexing**: Optimized indexes per category
- **Flexible Schema**: Each category can have category-specific fields
- **Parallel Queries**: Can search multiple categories in parallel

### Data Storage

- **Metadata**: Stored in PostgreSQL for fast retrieval
- **Raw Data**: CSV/Excel files loaded on-demand as pandas DataFrames
- **Embeddings**: 1536-dimensional vectors (OpenAI text-embedding-3-small)
- **Full-Text**: PostgreSQL tsvector for keyword search (legacy) or BM25 (current)

---

## Multi-Agent System

The system uses a four-agent pipeline orchestrated by LangGraph. Each agent is a specialized LangGraph workflow with its own nodes and edges.

### Agent Orchestrator

**File**: `backend/app/services/agents/orchestrator.py`

The orchestrator manages the overall workflow:

```python
Verification → [valid?] → Coordinator → Extraction → Analytics → END
                  ↓ [invalid]
                 END
```

**Key Responsibilities**:
- Initialize all agents with appropriate LLM configurations
- Route between agents based on validation results
- Collect agent responses and merge state
- Handle errors and early termination

**Agent-Specific LLM Configuration**:
- **Verification Agent**: Uses Haiku (fast, cheap) for simple validation
- **Analytics Agent**: Uses Opus 4.6 (most capable) for complex code generation
- **Other Agents**: Use default Sonnet 4.5 (balanced)

---

### 1. Query Verification Agent

**File**: `backend/app/services/agents/verification/agent.py`

**Purpose**: Validates user queries before processing to ensure they are relevant and have required information.

#### Workflow

```
validate_topic → [valid?] → extract_years → [has_year?] → check_dimensions
     ↓ [invalid]                  ↓ [missing]                    ↓
format_result                format_result            check_year_availability
                                                               ↓
                                                        format_result → END
```

#### Nodes

1. **validate_topic**: Checks if query is about employment, income, or hours worked
   - Uses keyword matching against `ALLOWED_TOPICS`
   - Returns `valid=True/False` with reason

2. **extract_years**: Extracts year or year range from query
   - Regex patterns: `\b(19\d{2}|20[0-4]\d)\b` for single years
   - Range patterns: "2019 to 2023", "between 2019 and 2023"
   - Returns `year_start`, `year_end`, or `needs_input=True`

3. **check_dimensions**: Identifies required dimensions (age, sex, industry, qualification)
   - Uses keyword matching for each dimension
   - Stores dimension flags in state for later filtering

4. **check_year_availability**: Validates requested years against dataset metadata
   - Queries database for available year ranges per category
   - Returns warning if requested year not fully available

5. **format_result**: Packages validation result as `QueryValidationResult` Pydantic model

#### Output

```python
{
    "query_validation": {
        "valid": True,
        "reason": None,
        "needs_input": False,
        "missing_info": None,
        "suggested_questions": [],
        "query": "Show me employment data for 2023",
        "year_range": {"start": 2023, "end": 2023},
        "dimensions_needed": {...}
    },
    "available_years": {
        "employment": {"min": 2010, "max": 2023},
        "income": {"min": 2012, "max": 2023}
    }
}
```

#### Example Validation Cases

- ❌ **Invalid Topic**: "What's the weather in Singapore?" → `valid=False`
- ❌ **Missing Year**: "Show me employment data" → `needs_input=True`
- ✅ **Valid**: "Employment in 2023 by age and sex" → `valid=True`

---

### 2. Data Coordinator Agent

**File**: `backend/app/services/agents/coordinator/agent.py`

**Purpose**: Plans the research workflow and delegates tasks to specialized agents.

#### Workflow

```
analyze_query → identify_data_sources → create_plan → determine_delegation → END
```

#### Nodes

1. **analyze_query**: Understands user intent using LLM
   - Identifies main intent (trend analysis, comparison, etc.)
   - Determines data type needed (employment, income, hours)
   - Extracts time scope and filters

2. **identify_data_sources**: Determines required datasets/APIs
   - Uses validation context (dimensions, years) from Verification Agent
   - Returns list of required data sources (e.g., `["employment", "income"]`)

3. **create_plan**: Builds workflow plan as list of steps
   - Creates `WorkflowStep` objects with agent, action, parameters
   - Example: `[{agent: "extraction", action: "retrieve_data", params: {...}}]`

4. **determine_delegation**: Decides which agent to call next
   - Usually delegates to Extraction Agent
   - Can skip directly to Analytics if data already available

#### Output

```python
{
    "workflow_plan": [
        {
            "agent": "extraction",
            "action": "retrieve_data",
            "parameters": {
                "categories": ["employment"],
                "year_filter": {"start": 2023, "end": 2023},
                "dimensions": ["age", "sex"]
            }
        }
    ],
    "metadata": {
        "required_data": ["employment"],
        "analysis_type": "trend_analysis"
    }
}
```

---

### 3. Data Extraction Agent

**File**: `backend/app/services/agents/extraction/agent.py`

**Purpose**: Extracts relevant data from datasets using RAG or file-based fallback.

#### Workflow

```
retrieve_context → load_dataframes → [has_data?] → extract_relevant_data → format_output → END
                                          ↓ [no_data]
                                     format_output → END
```

#### Nodes

1. **retrieve_context**: Uses RAG to find relevant datasets
   - Calls `RAGService.retrieve()` with query and filters
   - Gets back `TableSchema` objects with file paths and scores
   - Falls back to keyword-based file matching if RAG unavailable
   - Applies confidence-based filtering (default threshold: 0.5)

2. **load_dataframes**: Loads raw data from files as pandas DataFrames
   - Loads top 1-3 datasets based on confidence scores
   - Always loads top-ranked dataset (score >= threshold)
   - Loads additional datasets if score >= `confidence_threshold`
   - Respects `min_datasets` and `max_datasets` config

3. **extract_relevant_data**: Filters data based on query requirements (OPTIONAL)
   - This node can be skipped if full datasets are sufficient
   - Applies dimension filters (age, sex, industry, qualification)
   - Year filtering already done by RAG

4. **format_output**: Packages data for Analytics Agent
   - Serializes DataFrames: `{columns, dtypes, data, metadata, source: "dataframe"}`
   - Stores in `extracted_data` dict keyed by dataset name

#### Confidence-Based Dataset Selection

The extraction agent uses reranker scores to decide which datasets to load:

```python
# Example with 3 candidates and threshold=0.5:
# Dataset A: score=0.82 → ✅ Load (top 1, always loaded)
# Dataset B: score=0.61 → ✅ Load (above threshold)
# Dataset C: score=0.43 → ❌ Skip (below threshold)
```

**Logic**:
- Always load top 1 dataset (even if score < threshold, fallback safety)
- Load additional datasets if `score >= confidence_threshold`
- Guarantee at least `min_datasets` (default: 1)
- Never exceed `max_datasets` (default: 3)

**Log Output**:
```
✅ LOAD dataset_1.csv (score=0.82, rank=1)
✅ LOAD dataset_2.csv (score=0.61, rank=2, above threshold)
❌ SKIP dataset_3.csv (score=0.43, rank=3, below threshold 0.5)
```

#### Output

```python
{
    "extracted_data": {
        "mrsd_2024_employment.csv": {
            "columns": ["age", "sex", "employed_2023"],
            "dtypes": {"age": "object", "sex": "object", "employed_2023": "int64"},
            "data": [[...], [...], ...],  # List of rows
            "metadata": {
                "description": "Employment data by age and sex",
                "year_range": {"min": 2023, "max": 2023},
                "row_count": 120,
                "score": 0.82  # Relevance score from reranker
            },
            "source": "dataframe"
        }
    },
    "retrieval_context": {
        "table_schemas": [...]  # TableSchema objects with scores
    }
}
```

---

### 4. Analytics Agent

**File**: `backend/app/services/agents/analytics/agent.py`

**Purpose**: Analyzes data and generates insights with visualizations using pandas code generation.

#### Workflow

```
prepare_data → generate_code → execute_code → explain_results
                                     ↓
                          [needs_viz?] → generate_visualization
                                     ↓
                             compose_response → END
```

#### Nodes

1. **prepare_data**: Reconstructs DataFrames from serialized data
   - Deserializes column types (handles pandas datetime64, Int64, etc.)
   - Creates DataFrame objects in memory
   - Validates data availability

2. **generate_code**: Uses LLM to generate pandas analysis code
   - Uses Claude Opus 4.6 for chain-of-thought reasoning
   - Temperature: 0.0 (deterministic code generation)
   - Max tokens: 8192 (supports complex multi-step analysis)
   - Generates safe pandas code (no file I/O, network, or dangerous operations)

3. **execute_code**: Safely executes generated code
   - 5-second timeout to prevent infinite loops
   - Restricted environment (only pd, np, matplotlib allowed)
   - No file system or network access
   - Captures execution result (DataFrame, Series, or dict)

4. **explain_results**: LLM explains the analysis results
   - Converts technical results to natural language
   - Provides insights and key findings
   - Suggests follow-up questions

5. **generate_visualization** (conditional): Creates chart specification
   - Determines if visualization would enhance understanding
   - Generates `VisualizationData` Pydantic model
   - Supports: bar, line, pie, scatter charts
   - Extracts data from matplotlib figures or execution results

6. **compose_response**: Packages final response
   - Combines explanation, results, and visualization
   - Returns `AnalysisResult` Pydantic model

#### Code Generation Example

**User Query**: "Show me employment trends by age in 2023"

**Generated Code**:
```python
# Group by age and sum employment
df_grouped = df.groupby('age')['employed_2023'].sum().reset_index()
df_grouped = df_grouped.sort_values('age')

# Calculate statistics
total_employed = df_grouped['employed_2023'].sum()
max_age = df_grouped.loc[df_grouped['employed_2023'].idxmax(), 'age']

result = {
    'data': df_grouped.to_dict('records'),
    'total': int(total_employed),
    'peak_age': max_age
}
```

#### Code Execution Safety

**Allowed**:
- pandas operations (groupby, merge, pivot, etc.)
- numpy calculations
- matplotlib plotting
- Standard Python operations (if/else, loops, math)

**Blocked**:
- File I/O (`open()`, `pd.read_csv()`, etc.)
- Network access (`requests`, `urllib`, etc.)
- System calls (`os.system()`, `subprocess`, etc.)
- Dangerous builtins (`exec()`, `eval()`, `__import__()`)

#### Output

```python
{
    "message": "Employment in 2023 totaled 3.2 million, with peak employment in the 25-34 age group...",
    "visualization": {
        "type": "bar",
        "title": "Employment by Age Group (2023)",
        "data": {
            "labels": ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            "datasets": [{
                "label": "Employed",
                "data": [450000, 980000, 820000, 640000, 280000, 30000]
            }]
        },
        "options": {
            "indexAxis": "x",
            "plugins": {"legend": {"display": True}}
        }
    },
    "analysis_results": {
        "total_employed": 3200000,
        "peak_age_group": "25-34",
        "insights": [...]
    }
}
```

---

### Agent Communication

Agents communicate via shared `GraphState` (TypedDict):

```python
class GraphState(TypedDict):
    messages: List[BaseMessage]                  # Conversation history
    current_task: str                            # User's query
    extracted_data: Dict[str, Any]               # Serialized DataFrames
    analysis_results: Dict[str, Any]             # Analytics output
    workflow_plan: List[Dict[str, Any]]          # Coordinator's plan
    current_step: int                            # Workflow progress
    errors: List[str]                            # Error accumulation
    metadata: Dict[str, Any]                     # Shared context
    intermediate_results: Dict[str, Any]         # Agent responses
    should_continue: bool                        # Continue workflow?
    retrieval_context: Dict[str, Any]            # RAG results
    query_validation: Dict[str, Any]             # Verification result
    available_years: Dict[str, Dict[str, int]]   # Year ranges
```

**State Flow**:
1. Orchestrator creates initial `GraphState` from user message
2. Each agent receives state, executes its graph, returns updated state
3. Orchestrator merges agent responses back into shared state
4. Next agent receives cumulative state

**Key Pattern**: Verification agent stores validation data in `response.data["validation"]`, not in `state.to_graph_state()`. Orchestrator extracts this and merges into GraphState.

---

## RAG System

**File**: `backend/app/services/rag_service.py`

The RAG system uses hybrid search with BM25 and cross-encoder reranking for optimal retrieval.

### Hybrid Retrieval Pipeline

```
Query → [Vector Search] + [BM25 Search] → RRF Fusion → Cross-Encoder Reranking → Top-K Results
           (Top 20)          (Top 20)         (10)              (Re-scored)            (3-5)
```

### Components

#### 1. Vector Search (Semantic)

- **Model**: OpenAI text-embedding-3-small (1536 dimensions)
- **Index**: IVFFlat (pgvector) with cosine distance
- **Top-K**: 20 candidates
- **Scoring**: `similarity = 1.0 - distance`

```sql
SELECT *, embedding <=> '[...]'::vector AS distance
FROM employment_dataset_metadata
WHERE embedding IS NOT NULL
ORDER BY distance
LIMIT 20
```

#### 2. BM25 Search (Keyword)

- **Algorithm**: BM25Okapi (Okapi BM25 variant)
- **Library**: `rank-bm25==0.2.2`
- **Corpus**: `summary_text` field (rich dataset descriptions)
- **Top-K**: 20 candidates
- **Caching**: Two-tier cache (in-memory + disk pickle)

**Why BM25 over PostgreSQL Full-Text?**
- Better ranking algorithm (TF-IDF with length normalization)
- More predictable results (no stemming surprises)
- Tunable parameters (k1, b)
- Faster for small corpora (pre-computed indexes)

**Cache Strategy**:
```python
# Memory cache: {category_table: (BM25Okapi, rows)}
# Disk cache: ./data/bm25_cache/{category_table}.pkl

# Startup: Load all .pkl files into memory (warmup)
# Query: Use cached index for instant scoring
# Ingestion: Clear cache to rebuild with new data
```

#### 3. RRF Fusion (Reciprocal Rank Fusion)

Combines vector and BM25 results using reciprocal rank:

```python
score(doc) = Σ(1 / (k + rank(doc)))

# Example:
# Vector: doc_A rank 1, doc_B rank 5
# BM25:   doc_A rank 3, doc_C rank 1
# RRF (k=60):
#   doc_A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
#   doc_B: 1/(60+5)                               = 0.0154
#   doc_C:            1/(60+1)                    = 0.0164
# Final: doc_A > doc_C > doc_B
```

**Configuration**: `rrf_k=60` (balances vector and text results)

#### 4. Cross-Encoder Reranking

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (HuggingFace)
- **Size**: 80MB (fast inference)
- **Input**: Query + document pairs
- **Output**: Relevance score (0-1)
- **Execution**: Thread pool to avoid blocking event loop

**Why Reranking?**
- Considers query-document interactions (not just independent embeddings)
- More accurate than cosine similarity alone
- Corrects errors from initial retrieval
- MS MARCO trained on real search queries

**Async Execution**:
```python
# Offload CPU-bound reranking to thread pool
loop = asyncio.get_event_loop()
ranked_indices_scores = await loop.run_in_executor(
    self._executor,
    reranker.rerank,
    query,
    documents,
    top_k
)
```

### Filtering

**Category Filter** (optional):
- Narrows search to specific categories (employment, income, hours_worked)
- Example: Query mentions "salary" → filter to `income` category

**Year Filter** (from Verification Agent):
- Filters datasets by year range overlap
- Example: Query asks for 2023 → only load datasets with `year_range` covering 2023

### Retrieval Result

```python
FolderRetrievalResult(
    query="employment trends in 2023",
    metadata_results=[
        MetadataResult(
            metadata_id=123,
            category="employment",
            file_name="mrsd_2024_employment.csv",
            file_path="/path/to/file.csv",
            table_name="mrsd_2024_employment",
            description="...",
            columns=[...],
            year_range={"min": 2023, "max": 2023},
            score=0.82  # Reranker score
        ),
        # ... more results
    ],
    table_schemas=[
        TableSchema(
            table_name="mrsd_2024_employment",
            category="employment",
            file_path="/path/to/file.csv",
            score=0.82,  # Propagated to extraction agent
            ...
        )
    ],
    total_results=15
)
```

### Performance Optimizations

1. **Pre-computed Embeddings**: Generated during ingestion, not at query time
2. **BM25 Disk Cache**: Persists indexes to avoid cold-start latency
3. **Parallel Category Search**: Searches multiple categories concurrently
4. **IVFFlat Indexes**: Fast approximate nearest neighbor search
5. **Thread Pool Reranking**: Prevents blocking event loop
6. **Warmup on Startup**: Pre-loads BM25 caches from disk

---

## LLM Resilience & Fallback

**File**: `backend/app/services/llm_resilience.py`

Automatic retry, model fallback, and provider fallback to ensure reliable LLM access.

### Architecture

```
ResilientLLMService
  ├─ RetryStrategy (transient errors)
  ├─ ModelFallbackStrategy (token limits, model issues)
  ├─ ProviderFallbackStrategy (provider outages)
  └─ CircuitBreaker (prevent cascading failures)
```

### Error Classification

**TransientLLMError** (Retry):
- Rate limits (429)
- Network timeouts
- Server errors (500, 502, 503)

**PermanentLLMError** (Fallback):
- Authentication errors (401)
- Invalid model (404)
- General API failures

**TokenLimitError** (Model Fallback):
- Context length exceeded
- Max tokens exceeded

### Retry Strategy

**Algorithm**: Exponential backoff with jitter

```python
retry_config = {
    "max_attempts": 3,
    "initial_delay": 1.0,       # seconds
    "backoff_multiplier": 2.0,
    "max_delay": 30.0,
    "jitter": True,             # Prevent thundering herd
    "jitter_range": [0.0, 0.5]  # 0-50% of delay
}

# Attempt 1: Immediate
# Attempt 2: 1s + jitter(0-0.5s) = 1.0-1.5s
# Attempt 3: 2s + jitter(0-1.0s) = 2.0-3.0s
```

**Implementation**: Uses `tenacity` library

```python
@retry(
    retry=retry_if_exception_type(TransientLLMError),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=30, jitter=0.5),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
```

### Model Fallback

Progressively tries more expensive/capable models:

**Anthropic Chain**:
1. claude-sonnet-4-5-20250929 (base model, balanced)
2. claude-opus-4-6 (most capable, most expensive)

**OpenAI Chain**:
1. gpt-3.5-turbo (cheapest, fastest)
2. gpt-4 (mid-tier)
3. gpt-4-turbo-preview (most capable, largest context)

**Google Chain**:
1. gemini-pro (default)

**Agent-Specific Overrides**:
- **Analytics Agent**: Starts with Opus (needs code generation capability)
- **Verification Agent**: No fallback (single attempt with fast model)

### Provider Fallback

Tries providers in order:

1. **Anthropic** (primary, best for reasoning)
2. **OpenAI** (fallback, good embeddings)
3. **Google** (last resort)

**Configuration**:
```yaml
llm_fallback:
  provider_chain:
    - anthropic
    - openai
    - google
```

### Circuit Breaker

Prevents repeated calls to failing providers:

**States**:
- **CLOSED**: Normal operation, all calls allowed
- **OPEN**: Provider failing, block all calls (fail fast)
- **HALF_OPEN**: Testing recovery, allow limited calls

**Transitions**:
```
CLOSED → [5 failures] → OPEN → [60s timeout] → HALF_OPEN → [2 successes] → CLOSED
                                                     ↓ [failure]
                                                   OPEN
```

**Configuration**:
```yaml
circuit_breaker:
  failure_threshold: 5       # Open after 5 consecutive failures
  success_threshold: 2       # Close after 2 consecutive successes
  timeout: 60                # Seconds before attempting recovery
  half_open_max_calls: 3     # Max concurrent calls in half-open
```

### Integration with Agents

All agents use `BaseAgent._invoke_llm(enable_fallback=True)`:

```python
# In agent node:
response_content = await self._invoke_llm(
    messages=messages,
    include_system_prompt=True,
    enable_fallback=True  # Uses ResilientLLMService
)
```

**Fallback Flow**:
1. Try primary model (e.g., Sonnet) with retry
2. If all retries fail, try next model (e.g., Opus)
3. If all models fail, try next provider (OpenAI)
4. If all providers fail, raise `AllProvidersFailedError`

### Observability

**Log Events**:
- `[LLM RETRY]`: Retry attempt with delay
- `[LLM FALLBACK]`: Model or provider fallback
- `[CIRCUIT BREAKER]`: State changes (OPEN, HALF_OPEN, CLOSED)

**Example Logs**:
```
[LLM RETRY] Attempt 2/3 for claude-sonnet (rate_limit_error), retrying in 1.2s
[LLM FALLBACK] Model fallback: claude-sonnet → claude-opus (token_limit_exceeded)
[CIRCUIT BREAKER] OpenAI circuit OPEN (5 failures), blocking calls for 60s
[LLM FALLBACK] Provider fallback: OpenAI (circuit open) → Google
```

### Configuration

**Enable/Disable**:
```yaml
llm_fallback:
  enabled: true  # Master switch
```

**Per-Agent Overrides**:
```yaml
agent_overrides:
  analytics:
    model_chains:
      anthropic:
        - claude-opus-4-6        # Start with Opus
        - claude-sonnet-4-5
  verification:
    retry:
      max_attempts: 1            # No retry for fast validation
```

---

## Frontend Architecture

**Tech Stack**: React + TypeScript + Vite + Tailwind CSS

### Key Components

**1. Chat Interface** (`ChatInterface.tsx`)
- Message list with user/assistant bubbles
- Input box with send button
- Loading states and error handling
- Streaming message support

**2. Visualization Panel** (`VisualizationPanel.tsx`)
- Recharts library for chart rendering
- Supports: bar, line, pie, scatter charts
- Responsive design with Tailwind CSS

**3. Zustand Store** (`chatStore.ts`)
- Global state management
- Actions: `sendMessage()`, `clearMessages()`, `setVisualization()`
- Streaming state: `isStreaming`, `streamingMessage`

### API Client

**File**: `frontend/src/api/client.ts`

```typescript
// Chat endpoint (supports SSE streaming)
chatApi.sendMessage(content: string, conversationId?: string)

// Health check
configApi.checkHealth()
```

### Streaming Flow

1. User sends message via chat input
2. Frontend calls `/api/chat` with SSE headers
3. Backend streams response chunks as SSE events:
   - `agent`: Agent name update
   - `message`: Partial message chunk
   - `visualization`: Visualization spec
   - `done`: Final message
4. Frontend updates UI progressively as chunks arrive
5. Final message shown with visualization (if any)

---

## Design Decisions

### 1. Code Generation Instead of SQL

**Decision**: Use pandas code generation, not SQL generation.

**Rationale**:
- **Safety**: Easier to sandbox (no database access, limited environment)
- **Flexibility**: Pandas supports complex transformations SQL can't express easily
- **Portability**: Works with any data source (CSV, Excel, JSON), not just databases
- **Debugging**: Python code is easier to inspect and debug than SQL
- **Visualization**: Direct integration with matplotlib for charts

**Trade-offs**:
- Performance: Slower for very large datasets (SQL is database-optimized)
- Memory: Loads entire dataset into memory (SQL streams)
- Accepted because government datasets are small (<100MB)

### 2. Folder-Based Metadata Tables

**Decision**: Separate metadata tables per category, not a single unified table.

**Rationale**:
- **Scalability**: Avoids single table bottleneck as data grows
- **Parallel Queries**: Can search multiple categories concurrently
- **Domain-Specific Fields**: Each category can have custom metadata
- **Index Optimization**: Smaller indexes, faster queries

**Trade-offs**:
- Complexity: More tables to manage, more code for ingestion
- Duplication: Schema repeated across tables
- Accepted because benefits outweigh maintenance cost

### 3. BM25 Instead of PostgreSQL Full-Text

**Decision**: Use rank-bm25 library with disk caching, not PostgreSQL tsvector.

**Rationale**:
- **Better Ranking**: BM25 is state-of-the-art for keyword matching (TF-IDF + length normalization)
- **Predictable**: No stemming surprises (PostgreSQL's stemmer can be aggressive)
- **Fast for Small Corpora**: Pre-computed indexes, no query parsing overhead
- **Tunable**: Can adjust k1, b parameters for domain

**Trade-offs**:
- Memory: Requires loading corpus into memory (cached)
- Complexity: Two-tier cache system (memory + disk)
- Accepted because datasets are small and warmup is fast

### 4. Confidence-Based Dataset Loading

**Decision**: Load 1-3 datasets based on reranker scores, not all top-k.

**Rationale**:
- **Efficiency**: Avoid loading irrelevant data (faster, less memory)
- **Quality**: Focus analysis on most relevant datasets (better results)
- **Cost**: Less data to process → cheaper LLM calls
- **Tunable**: `confidence_threshold` can be adjusted per deployment

**Trade-offs**:
- Risk of missing data: If threshold too high, might skip relevant datasets
- Complexity: More logic in extraction agent
- Accepted because fallback guarantees min_datasets=1

### 5. Multi-Agent with LangGraph

**Decision**: Use four specialized agents, not a single monolithic agent.

**Rationale**:
- **Separation of Concerns**: Each agent has clear responsibility
- **Reusability**: Agents can be tested and developed independently
- **Flexibility**: Can add/remove agents without affecting others
- **Observability**: Clear trace of which agent did what
- **Chain-of-Thought**: Each agent's reasoning is explicit

**Trade-offs**:
- Complexity: More orchestration code, more state management
- Latency: Sequential agent calls add overhead
- Accepted because benefits in maintainability and debuggability

### 6. LLM Fallback with Circuit Breaker

**Decision**: Automatic retry, model fallback, provider fallback with circuit breaker.

**Rationale**:
- **Reliability**: Tolerate transient provider issues (rate limits, timeouts)
- **Cost Optimization**: Try cheaper models first, fall back to expensive only if needed
- **Fail Fast**: Circuit breaker prevents wasting time on known-bad providers
- **User Experience**: Seamless failover, no manual intervention

**Trade-offs**:
- Complexity: More code, more configuration
- Latency: Retries and fallbacks add delay
- Accepted because production systems need resilience

### 7. OpenAI for Embeddings, Anthropic for LLM

**Decision**: Use OpenAI text-embedding-3-small for embeddings, Anthropic Claude for LLM.

**Rationale**:
- **Embeddings**: OpenAI has best-in-class small embeddings (1536d, cheap, fast)
- **LLM**: Anthropic Claude excels at reasoning and code generation
- **Separation**: Embedding and generation are independent concerns

**Trade-offs**:
- Vendor Lock-in: Depends on two providers
- Cost: Two API bills
- Accepted because each provider is best at their task

### 8. Streaming with SSE

**Decision**: Use Server-Sent Events (SSE), not WebSockets.

**Rationale**:
- **Simplicity**: Unidirectional (server→client), no bidirectional complexity
- **HTTP-Friendly**: Works over HTTP/1.1, no protocol upgrade
- **Browser Support**: Native EventSource API
- **Reconnection**: Automatic reconnection on disconnect

**Trade-offs**:
- Unidirectional: Can't push updates from client during streaming
- Not Supported in All Browsers: (but works in all modern browsers)
- Accepted because our use case is server→client only

---

## Data Flow

### End-to-End Query Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. User Input                                                    │
│    "Show me employment trends in 2023 by age"                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Verification Agent                                            │
│    ✓ Valid topic (employment)                                   │
│    ✓ Year extracted (2023)                                      │
│    ✓ Dimensions (age)                                           │
│    ✓ Year available in datasets                                 │
│    → validation = {valid: true, year_range: {2023, 2023}}       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Coordinator Agent                                             │
│    • Analyze query: trend analysis, employment data, 2023       │
│    • Identify sources: employment datasets                       │
│    • Create plan: [extract employment, analyze trends]          │
│    → workflow_plan = [...]                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Extraction Agent                                              │
│    A. retrieve_context (RAG)                                    │
│       • Embed query: [0.123, -0.456, ...] (1536d)              │
│       • Vector search: Top 20 by cosine similarity              │
│       • BM25 search: Top 20 by keyword matching                 │
│       • RRF fusion: Combine to 10 results                       │
│       • Rerank: Cross-encoder scores                            │
│       • Filter: year=2023, category=employment                  │
│       → table_schemas = [                                       │
│           {file_path: "mrsd_2024.csv", score: 0.82},           │
│           {file_path: "employment_by_age.csv", score: 0.61}    │
│         ]                                                        │
│                                                                  │
│    B. load_dataframes                                           │
│       • Load top 1: mrsd_2024.csv (score=0.82, always)         │
│       • Load #2: employment_by_age.csv (score=0.61 >= 0.5)    │
│       → loaded_datasets = {                                     │
│           "mrsd_2024.csv": DataFrame(...),                     │
│           "employment_by_age.csv": DataFrame(...)              │
│         }                                                        │
│                                                                  │
│    C. format_output                                             │
│       • Serialize DataFrames: {columns, dtypes, data, ...}     │
│       → extracted_data = {...}                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Analytics Agent                                               │
│    A. prepare_data                                              │
│       • Reconstruct DataFrames from serialized data             │
│       → df1 = pd.DataFrame(...), df2 = pd.DataFrame(...)       │
│                                                                  │
│    B. generate_code (LLM: Claude Opus 4.6)                      │
│       • Prompt: "Analyze employment trends by age in 2023"      │
│       • Generated code:                                          │
│         ```python                                                │
│         df_combined = pd.concat([df1, df2])                      │
│         df_trends = df_combined.groupby('age')['employed'].sum() │
│         result = df_trends.to_dict()                             │
│         ```                                                      │
│                                                                  │
│    C. execute_code                                              │
│       • Run in sandbox (timeout=5s, restricted env)             │
│       → result = {                                              │
│           '15-24': 450000, '25-34': 980000, ...                │
│         }                                                        │
│                                                                  │
│    D. explain_results (LLM)                                     │
│       • Prompt: "Explain these employment trends"               │
│       → message = "Employment in 2023 peaked in 25-34 age..."  │
│                                                                  │
│    E. generate_visualization                                    │
│       → visualization = {                                        │
│           type: "bar",                                          │
│           title: "Employment by Age (2023)",                    │
│           data: {labels: [...], datasets: [...]}               │
│         }                                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Response to User                                              │
│    {                                                             │
│      message: "Employment in 2023 peaked in 25-34 age...",     │
│      visualization: {type: "bar", ...},                         │
│      sources: ["mrsd_2024.csv", "employment_by_age.csv"],      │
│      agent_trace: ["Verification", "Coordinator", ...]          │
│    }                                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. Frontend Rendering                                            │
│    • Display message in chat bubble                             │
│    • Render bar chart with Recharts                             │
│    • Show agent trace in metadata                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

This architecture provides a robust, scalable, and intelligent system for querying Singapore government datasets. The multi-agent design ensures clear separation of concerns, while the hybrid RAG system with BM25 and reranking delivers high-quality retrieval. The LLM fallback mechanism ensures reliability in production, and the code generation approach provides safety and flexibility.

**Key Strengths**:
- 🧠 **Intelligent**: Multi-agent reasoning with specialized roles
- 🔍 **Accurate**: Hybrid RAG with reranking (precision over recall)
- 🛡️ **Resilient**: Automatic retry and fallback with circuit breaker
- 🔒 **Safe**: Sandboxed code execution with restricted environment
- 📊 **Visual**: Rich visualizations generated from analysis
- ⚡ **Fast**: Confidence-based loading, BM25 caching, parallel queries

**Future Improvements**:
- Add more data sources (weather, traffic, health)
- Support cross-category queries (employment + income correlation)
- Implement query caching for repeated questions
- Add user authentication and query history
- Expand visualization types (maps, networks, etc.)
