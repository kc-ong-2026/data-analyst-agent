# Singapore Government Data Chat Assistant

An intelligent AI-powered chat assistant that enables natural language querying and analysis of Singapore government datasets with real-time visualization capabilities. Built with a multi-agent architecture using LangGraph, hybrid RAG retrieval, and resilient LLM infrastructure.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack & Justifications](#technology-stack-justifications)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)
- [Sample Queries](#sample-queries)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

<a name="overview"></a>
## 🎯 Overview

An AI-powered chat assistant for natural language querying of Singapore government datasets (employment, income, hours worked). Ask questions in plain English and receive intelligent insights with auto-generated visualizations.

**Key Capabilities**: Multi-agent pipeline validates queries → retrieves data via hybrid RAG (vector + BM25 search) → generates pandas code for analysis → creates visualizations.

**Read [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.**

---

<a name="key-features"></a>
## ✨ Key Features

- 🤖 **Natural Language Queries** - Ask in plain English (e.g., "Show me employment trends in 2023 by age")
- 📊 **Auto-Generated Visualizations** - Bar, line, pie, and scatter charts
- 🔍 **Hybrid RAG Retrieval** - Vector + BM25 search with cross-encoder reranking
- 🛡️ **LLM Resilience** - Automatic retry, model fallback, provider fallback with circuit breaker
- 🏗️ **Multi-Agent Architecture** - Verification → Coordinator → Extraction → Analytics pipeline
- 🔒 **Secure Code Execution** - AST-based validation, sandboxed execution, comprehensive audit logging
- 🔄 **Provider Agnostic** - Supports OpenAI, Anthropic, and Google models
- 🐳 **Docker Ready** - Full containerization with hot reload for development
- 🧪 **Comprehensive Testing** - 75+ tests across backend (pytest) and frontend (Jest) with 90%+ coverage

---

<a name="architecture"></a>
## 🏛️ Architecture

```
User Query
    ↓
Verification Agent (validates topic & year)
    ↓
Coordinator Agent (creates execution plan)
    ↓
Extraction Agent (RAG: Vector + BM25 + Reranking)
    ↓
Analytics Agent (pandas code generation + visualization)
    ↓
Response + Chart
```

**Stack**: React + TypeScript → FastAPI + LangGraph → PostgreSQL + pgvector

**📖 For detailed architecture, design decisions, and system diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md)**

---

<a name="technology-stack-justifications"></a>
## 🛠️ Technology Stack & Justifications

### Frontend

| Technology | Version | Why We Use It |
|-----------|---------|---------------|
| **React** | 18.x | Industry-standard UI library with great ecosystem and performance |
| **TypeScript** | 5.x | Type safety prevents runtime errors and improves developer experience |
| **Vite** | 5.x | Blazing fast dev server and build tool with HMR |
| **Tailwind CSS** | 3.x | Utility-first CSS for rapid UI development without CSS bloat |
| **Recharts** | 2.x | React-native charting library with declarative API |
| **Zustand** | 4.x | Minimal state management with less boilerplate than Redux |
| **Axios** | 1.x | Robust HTTP client with interceptors and better error handling |
| **Jest** | 29.x | Comprehensive testing framework with great TypeScript support |
| **Testing Library** | 14.x | Best practices for testing React components and hooks |

### Backend

| Technology | Version | Why We Use It |
|-----------|---------|---------------|
| **Python** | 3.11 | Latest stable Python with better performance and type hints |
| **FastAPI** | 0.104+ | Modern async framework with auto-generated OpenAPI docs |
| **LangChain** | 0.1.x | LLM framework with provider abstractions and tools |
| **LangGraph** | 0.0.x | State machine for multi-agent workflows with built-in orchestration |
| **Pydantic** | 2.x | Data validation with excellent TypeScript integration |
| **SQLAlchemy** | 2.x | ORM with async support and migration tools |
| **PostgreSQL** | 16 | Robust relational database with excellent JSON support |
| **pgvector** | 0.5.x | Native vector similarity search in PostgreSQL |

### AI/ML Stack

| Technology | Why We Use It |
|-----------|---------------|
| **Anthropic Claude** | Best-in-class reasoning and code generation (Opus 4.6 for analytics) |
| **OpenAI GPT** | Excellent embeddings (text-embedding-3-small) and fallback LLM |
| **Google Gemini** | Secondary fallback provider for redundancy |
| **rank-bm25** | Better keyword search than PostgreSQL full-text (BM25Okapi algorithm) |
| **sentence-transformers** | Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) |
| **pandas** | Industry-standard data manipulation with rich API |
| **matplotlib** | Flexible visualization library for chart generation |

### Infrastructure

| Technology | Why We Use It |
|-----------|---------------|
| **Docker** | Containerization for consistent dev/prod environments |
| **Docker Compose** | Multi-container orchestration for local development |
| **Make** | Simplified command-line interface for common tasks |

### Key Design Decisions

1. **Pandas Code Generation (not SQL)**
   - Safer to sandbox (no database access)
   - More flexible for complex transformations
   - Easier to debug and inspect
   - Direct matplotlib integration

2. **Multi-Agent Architecture**
   - Clear separation of concerns
   - Independent testing and development
   - Better error handling and recovery
   - Observable execution traces

3. **Hybrid RAG (Vector + BM25)**
   - Vector search captures semantic meaning
   - BM25 excels at keyword matching
   - RRF fusion combines strengths
   - Cross-encoder reranking improves precision

4. **LLM Fallback Mechanism**
   - Resilience against rate limits and outages
   - Cost optimization (try cheaper models first)
   - Circuit breaker prevents wasted retries
   - Seamless failover for users

5. **Folder-Based Metadata Tables**
   - Scalable (avoids single table bottleneck)
   - Parallel queries across categories
   - Domain-specific optimizations
   - Better indexing performance

---

<a name="setup-instructions"></a>
## 🚀 Setup Instructions

### Prerequisites

Before starting, ensure you have:

- **Docker** (20.10+) and **Docker Compose** (2.0+)
- **API Keys** for at least one LLM provider:
  - [OpenAI API Key](https://platform.openai.com/api-keys) (Required for embeddings)
  - [Anthropic API Key](https://console.anthropic.com/) (Recommended for best results)
  - [Google AI API Key](https://makersuite.google.com/app/apikey) (Optional fallback)
- **Git** for cloning the repository
- At least **4GB RAM** available for Docker containers
- **10GB disk space** for Docker images and data

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/your-org/govtech-chat-assistant.git
cd govtech-chat-assistant
```

#### 2. Configure Environment Variables

Create the backend environment file:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and add your API keys:

```bash
# Required: OpenAI for embeddings
OPENAI_API_KEY=sk-...

# Recommended: Anthropic for best LLM performance
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Google as fallback provider
GOOGLE_API_KEY=...

# Database (default values work with docker-compose)
DATABASE_URL=postgresql+asyncpg://govtech:govtech_dev@postgres:5432/govtech_rag

# Optional: LangSmith tracing (for debugging)
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=govtech-multi-agent-system
```

**⚠️ Important**: The OpenAI API key is required even if you're using Anthropic for LLM, because we use OpenAI embeddings for RAG.

#### 3. Configure LLM Settings (Optional)

Edit `backend/config/config.yaml` to customize:

```yaml
llm:
  default_provider: anthropic  # or openai, google

  providers:
    anthropic:
      default_model: claude-sonnet-4-5-20250929
      temperature: 0.7
      max_tokens: 4096

      # Agent-specific models
      analytics_model: claude-opus-4-6  # Use Opus for analytics
      verification_model: claude-sonnet-4-5-20250929  # Fast model for verification

llm_fallback:
  enabled: true  # Enable automatic retry and fallback

rag:
  use_reranking: true  # Enable cross-encoder reranking
  use_bm25: true       # Use BM25 instead of PostgreSQL full-text
  confidence_threshold: 0.5  # Minimum score to load dataset
```

#### 4. Build and Start Services

**Option A: Production Mode** (optimized, no hot reload)

```bash
make build
make up
```

**Option B: Development Mode** (hot reload, better for development)

```bash
make dev
```

This will:
1. Build Docker images for frontend, backend, and database
2. Start PostgreSQL with pgvector extension
3. Start backend server on port 8000
4. Start frontend server on port 3000
5. Automatically ingest datasets into the database

#### 5. Verify Installation

Wait for all services to start (about 30-60 seconds), then check:

```bash
# Check service health
curl http://localhost:8000/api/config/health

# Expected response:
# {"status":"healthy","database":"connected","vector_store":"ready"}
```

#### 6. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)

---

<a name="running-the-application"></a>
## 🎮 Running the Application

### Docker (Recommended)

#### Production Mode

```bash
# Start all services
make up

# View logs
make logs

# View backend logs only
make logs-backend

# Stop services
make down

# Clean up (remove containers and volumes)
make clean
```

#### Development Mode (with hot reload)

```bash
# Start development containers
make dev

# This enables:
# - Frontend: Vite hot module replacement (HMR)
# - Backend: uvicorn auto-reload on file changes
# - Database: persistent volume for data
```

### Local Development (without Docker)

If you prefer running services locally:

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your API keys

# Start PostgreSQL (if not using Docker)
# Note: You need pgvector extension installed
# See: https://github.com/pgvector/pgvector#installation

# Run migrations (if needed)
alembic upgrade head

# Start backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Frontend will be available at http://localhost:5173
```

### Data Ingestion

The system automatically ingests datasets on first startup. To manually trigger ingestion:

```bash
# Via Docker
make ingest

# Or directly
docker-compose exec backend python -m app.services.ingestion.ingest_cli

# Local (without Docker)
cd backend
python -m app.services.ingestion.ingest_cli
```

**What happens during ingestion:**
1. Scans `dataset/` directory for CSV/Excel files
2. Extracts metadata (columns, year ranges, dimensions)
3. Generates embeddings using OpenAI text-embedding-3-small
4. Creates PostgreSQL full-text search vectors
5. Builds BM25 indexes and caches to disk
6. Creates IVFFlat vector indexes for fast similarity search

**Data ingestion takes 2-5 minutes** depending on dataset size and API rate limits.

### Database Access

```bash
# Open PostgreSQL shell
make db-shell

# Or directly
docker-compose exec postgres psql -U govtech -d govtech_rag

# Useful queries:
# SELECT COUNT(*) FROM employment_dataset_metadata;
# SELECT file_name, description FROM employment_dataset_metadata;
# \dt  -- List all tables
```

---

<a name="running-tests"></a>
## 🧪 Running Tests

The project includes comprehensive testing with unit tests, integration tests, security tests, and evaluation tests for both backend and frontend.

### Quick Test Commands

```bash
# Run all backend tests
make test

# Run all frontend tests
make test-frontend

# Run ALL tests (backend + frontend)
make test-all

# Run specific backend test categories
make test-unit              # Backend unit tests only (fast, ~30s)
make test-integration       # Integration tests with mocked LLM (fast)
make test-security          # Security tests (code validation, sandboxing)
make test-e2e              # End-to-end pipeline tests

# Run with coverage reports
make test-coverage          # Backend coverage
make test-frontend-coverage # Frontend coverage
```

### Detailed Test Commands

#### Unit Tests

Test individual components in isolation:

```bash
# Run all unit tests
make test-unit

# Run specific test file
make test-unit TEST=test_verification_agent.py

# Run specific test function
make test-unit TEST=test_verification_agent.py::test_valid_query

# Local (without Docker)
cd backend
pytest tests/unit/ -v
```

**Unit test files:**
- `test_verification_agent.py` - Query validation logic
- `test_coordinator_agent.py` - Workflow planning
- `test_extraction_agent.py` - Data retrieval
- `test_analytics_agent.py` - Code generation and analysis
- `test_rag_retrieval.py` - RAG search functionality

#### Integration Tests

Test agent interactions and workflows:

```bash
# Fast integration tests (mocked LLM, no DB)
make test-integration

# Integration tests with database (skip LLM)
make test-integration-with-db

# Full integration tests including LLM (SLOW, uses API credits)
make test-integration-full
```

#### Evaluation Tests

Measure system quality with metrics:

```bash
# Run all evaluation tests
make test-evaluation

# Test retrieval quality (Precision, Recall, MRR, NDCG)
make test-evaluation-retrieval

# Test generation quality (BLEU, ROUGE, BERTScore)
make test-evaluation-generation

# Export results for fine-tuning
make test-evaluation-export
```

#### Smoke Tests

Quick validation that core functionality works:

```bash
make test-smoke
```

#### Frontend Tests

Test frontend components, state management, and API client:

```bash
# Run all frontend tests
make test-frontend

# Run in watch mode (local, for development)
make test-frontend-watch

# Run with coverage report
make test-frontend-coverage

# Local (without Docker)
cd frontend
npm test                    # Run all tests
npm run test:watch          # Watch mode
npm run test:coverage       # With coverage
```

**Frontend test files:**
- `api/__tests__/client.test.ts` - API client, SSE streaming, HTTP requests (15 tests, 93% coverage)
- `store/__tests__/chatStore.test.ts` - Zustand state management, messages, visualization (12 tests, 98% coverage)
- `types/__tests__/index.test.ts` - TypeScript type definitions (8 tests, 100% coverage)

### Test Structure

```
backend/tests/                      # Backend tests (Python + pytest)
├── unit/                           # Unit tests (fast, isolated)
│   ├── test_verification_agent.py
│   ├── test_coordinator_agent.py
│   ├── test_extraction_agent.py
│   ├── test_analytics_agent.py
│   └── test_rag_retrieval.py
│
├── integration/                    # Integration tests (slower, real flows)
│   ├── test_e2e_pipeline.py
│   ├── test_orchestrator.py
│   └── test_rag_pipeline.py
│
├── security/                       # Security tests
│   ├── test_code_validator.py
│   ├── test_sandbox_executor.py
│   └── test_security_integration.py
│
├── evaluation/                     # Quality metrics
│   ├── test_retrieval_metrics.py
│   └── test_generation_metrics.py
│
└── test_smoke.py                  # Quick validation

frontend/src/                       # Frontend tests (TypeScript + Jest)
├── api/__tests__/
│   └── client.test.ts              # API client tests (15 tests)
├── store/__tests__/
│   └── chatStore.test.ts           # State management tests (12 tests)
└── types/__tests__/
    └── index.test.ts               # Type definition tests (8 tests)
```

### Running Tests Locally (without Docker)

```bash
cd backend

# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html --cov-report=term

# Open coverage report
open htmlcov/index.html
```

### Test Best Practices

- **Fast Feedback**: Unit tests should run in <30 seconds
- **Mocking**: Integration tests mock LLM calls by default (save API credits)
- **Isolation**: Each test is independent (no shared state)
- **Fixtures**: Use pytest fixtures for common setup
- **Markers**: Tests are marked with `@pytest.mark.requires_llm` if they need real LLM calls

### Code Quality & Linting

**Backend** (Python - Ruff + Black):
```bash
cd backend

# Install linting tools
pip install -r requirements-dev.txt

# Run Ruff linter
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code with Black
black .

# Check formatting without changes
black --check .
```

**Frontend** (TypeScript - ESLint):
```bash
cd frontend

# Run ESLint
npm run lint

# Auto-fix issues
npm run lint -- --fix

# Type check
npx tsc --noEmit
```

**CI/CD**: All linting checks run automatically in GitHub Actions on every push.

---

<a name="sample-queries"></a>
## 💬 Sample Queries

Try these queries based on the actual datasets available (1991-2025):

### Employment Rate Queries

```
1. "What is the employment rate for males vs females in 2023?"
   ✅ Dataset: Resident_Employment_Rate_By_Age_and_Sex.csv
   → Compares male/female employment rates by age groups

2. "Show me employment rate trends for people aged 55-64 from 2010 to 2023"
   ✅ Dataset: Resident_Employment_Rate_By_Age_and_Sex.csv
   → Line chart showing elderly employment trends over time

3. "What is the employment rate for females aged 25-54 in 2020?"
   ✅ Dataset: Resident_Employment_Rate_By_Age_and_Sex.csv
   → Specific demographic employment rate
```

### Employment by Education Level

```
4. "How many residents with primary education are employed in 2023?"
   ✅ Dataset: Employed_Residents_by_Highest_Qualification_Attained_and_Sex.csv
   → Shows employed counts by education level

5. "Compare employment between males and females with university degrees in 2020"
   ✅ Dataset: Employed_Residents_by_Highest_Qualification_Attained_and_Sex.csv
   → Gender comparison for specific education level
```

### Employment by Occupation

```
6. "Show me the number of professionals employed in 2023"
   ✅ Dataset: Number_of_Employed_Residents_by_Occupation_and_Sex.csv
   → Employment counts by occupation category

7. "Compare employment between managers and service workers in 2020"
   ✅ Dataset: Number_of_Employed_Residents_by_Occupation_and_Sex.csv
   → Cross-occupation comparison

8. "How many female clerical workers were employed in 2023?"
   ✅ Dataset: Number_of_Employed_Residents_by_Occupation_and_Sex.csv
   → Gender + occupation filter
```

### Income Analysis

```
9. "What is the median income in 2023?"
   ✅ Dataset: Gross_Monthly_Income_From_Employment_of_Full_Time_Employed_Residents.csv
   → Shows p50 (median) income including/excluding CPF

10. "Show income trends from 2010 to 2023"
    ✅ Dataset: Gross_Monthly_Income_From_Employment_of_Full_Time_Employed_Residents.csv
    → Line chart showing income growth over time

11. "What is the 20th percentile income in 2020?"
    ✅ Dataset: Gross_Monthly_Income_From_Employment_of_Full_Time_Employed_Residents.csv
    → Shows p20 income statistics
```

### Hours Worked Analysis

```
12. "What are the average working hours in Manufacturing in 2023?"
    ✅ Dataset: annual_average_Hours_Worked_Per_Employed_Person_Aged_Fifteen_And_Over_By_Industry.csv
    → Industry-specific hours worked

13. "Compare working hours between Retail Trade and Financial Services in 2020"
    ✅ Dataset: annual_average_Hours_Worked_Per_Employed_Person_Aged_Fifteen_And_Over_By_Industry.csv
    → Cross-industry hours comparison

14. "Show me working hours trends in Construction from 2017 to 2023"
    ✅ Dataset: annual_average_Hours_Worked_Per_Employed_Person_Aged_Fifteen_And_Over_By_Industry.csv
    → Time series for specific industry
```

### Employment Change by Industry

```
15. "How did employment in Construction change in 2023?"
    ✅ Dataset: Annual_Employment_Change_by_Industry_28042025.csv
    → Employment change statistics by industry

16. "Compare employment changes in Manufacturing vs Services in 2020"
    ✅ Dataset: Annual_Employment_Change_by_Industry_28042025.csv
    → Cross-sector comparison of employment changes
```

### Invalid Queries (Testing Validation)

```
❌ "What's the weather in Singapore?"
   Response: "I can only answer questions about Singapore employment, income, and hours worked data."

❌ "Show me employment data"
   Response: "Please specify which year or year range you're interested in."

❌ "How many people work in 1980?"
   Response: "Data for 1980 is not available. Available years: 1991-2025."

❌ "What is the GDP of Singapore?"
   Response: "I can only answer questions about employment, income, and hours worked."
```

### Expected Response Format

For each valid query, you'll receive:

1. **Natural Language Explanation**: AI-generated insights from the data
2. **Visualization** (if appropriate): Chart rendered by Recharts (bar/line/pie/scatter)
3. **Sources**: List of datasets used (e.g., "Resident_Employment_Rate_By_Age_and_Sex.csv")
4. **Agent Trace**: Shows pipeline steps (Verification → Coordinator → Extraction → Analytics)

Example response for "What is the employment rate for males aged 25-54 in 2023?":

```json
{
  "message": "In 2023, the employment rate for males aged 25-54 was 95.1%, showing strong labor force participation...",
  "visualization": {
    "type": "bar",
    "title": "Employment Rate by Age Group (2023)",
    "data": [...]
  },
  "sources": ["Resident_Employment_Rate_By_Age_and_Sex.csv"],
  "metadata": {
    "execution_time": "2.3s",
    "datasets_loaded": 1
  }
}
```

---

<a name="project-structure"></a>
## 📁 Project Structure

```
govtech-chat-assistant/
├── frontend/                          # React + TypeScript frontend
│   ├── src/
│   │   ├── components/                # React components
│   │   │   ├── ChatInterface.tsx      # Main chat UI
│   │   │   ├── VisualizationPanel.tsx # Chart rendering
│   │   │   └── MessageBubble.tsx      # Chat messages
│   │   ├── store/                     # Zustand state management
│   │   │   └── chatStore.ts           # Chat state and actions
│   │   ├── api/                       # API client
│   │   │   └── client.ts              # HTTP client with SSE
│   │   ├── types/                     # TypeScript types
│   │   │   └── index.ts               # Type definitions
│   │   ├── App.tsx                    # Root component
│   │   └── main.tsx                   # Entry point
│   ├── public/                        # Static assets
│   ├── Dockerfile                     # Frontend container
│   ├── package.json                   # NPM dependencies
│   └── vite.config.ts                 # Vite configuration
│
├── backend/                           # Python FastAPI backend
│   ├── app/
│   │   ├── routes/                    # API endpoints
│   │   │   ├── chat.py                # Chat endpoints (SSE)
│   │   │   ├── config.py              # Config endpoints
│   │   │   └── data.py                # Data endpoints
│   │   ├── services/                  # Business logic
│   │   │   ├── agents/                # Multi-agent system
│   │   │   │   ├── verification/      # Query validation
│   │   │   │   ├── coordinator/       # Workflow planning
│   │   │   │   ├── extraction/        # Data retrieval
│   │   │   │   ├── analytics/         # Analysis & viz
│   │   │   │   ├── orchestrator.py    # Agent orchestration
│   │   │   │   └── base_agent.py      # Base agent class
│   │   │   ├── security/              # Code execution security
│   │   │   │   ├── code_validator.py  # AST-based code validation
│   │   │   │   ├── sandbox_executor.py # Sandboxed code execution
│   │   │   │   ├── audit_logger.py    # Security audit logging
│   │   │   │   └── exceptions.py      # Security exceptions
│   │   │   ├── ingestion/             # Data ingestion pipeline
│   │   │   ├── llm_service.py         # LLM provider abstraction
│   │   │   ├── llm_resilience.py      # Retry + fallback logic
│   │   │   ├── rag_service.py         # Hybrid RAG retrieval
│   │   │   ├── reranker.py            # Cross-encoder reranking
│   │   │   └── data_service.py        # Data loading (DataFrames)
│   │   ├── models/                    # Pydantic models
│   │   ├── db/                        # Database models
│   │   │   └── models.py              # SQLAlchemy ORM
│   │   ├── config.py                  # App configuration
│   │   └── main.py                    # FastAPI app
│   ├── config/
│   │   └── config.yaml                # YAML configuration
│   ├── db/
│   │   └── init.sql                   # Database schema
│   ├── tests/                         # Test suite
│   │   ├── unit/                      # Unit tests
│   │   ├── integration/               # Integration tests
│   │   ├── security/                  # Security tests
│   │   └── evaluation/                # Quality metrics
│   ├── Dockerfile                     # Backend container
│   ├── requirements.txt               # Python dependencies
│   └── .env.example                   # Environment template
│
├── dataset/                           # Singapore government datasets
│   └── singapore_manpower_dataset/    # Employment, income, hours
│
├── api_spec/                          # API specifications
│   └── singapore_environment_api_specs/ # Weather, air quality
│
├── docker-compose.yml                 # Production compose
├── docker-compose.dev.yml             # Development compose
├── Makefile                           # CLI commands
├── ARCHITECTURE.md                    # Architecture documentation
├── CLAUDE.md                          # AI assistant instructions
└── README.md                          # This file
```

---

<a name="configuration"></a>
## ⚙️ Configuration

### Backend Configuration (`backend/config/config.yaml`)

#### LLM Configuration

```yaml
llm:
  default_provider: anthropic  # openai, anthropic, google

  providers:
    anthropic:
      models:
        - claude-opus-4-6
        - claude-sonnet-4-5-20250929
      default_model: claude-sonnet-4-5-20250929
      temperature: 0.7
      max_tokens: 4096

      # Agent-specific models
      verification_model: claude-sonnet-4-5-20250929  # Fast validation
      verification_temperature: 0.0
      verification_max_tokens: 1024

      analytics_model: claude-opus-4-6  # Best for code generation
      analytics_temperature: 0.0
      analytics_max_tokens: 8192
```

#### LLM Fallback Configuration

```yaml
llm_fallback:
  enabled: true

  # Retry configuration
  retry:
    max_attempts: 3
    initial_delay: 1.0
    backoff_multiplier: 2.0
    max_delay: 30.0

  # Circuit breaker
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    success_threshold: 2
    timeout: 60

  # Provider fallback chain
  provider_chain:
    - anthropic
    - openai
    - google
```

#### RAG Configuration

```yaml
rag:
  vector_search_top_k: 20        # Vector search candidates
  fulltext_search_top_k: 20      # BM25 candidates
  hybrid_top_k: 10               # After RRF fusion
  rrf_k: 60                      # RRF parameter

  use_reranking: true            # Cross-encoder reranking
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

  use_bm25: true                 # Use BM25 instead of PostgreSQL FTS

  confidence_threshold: 0.5      # Min score to load dataset
  min_datasets: 1                # Always load at least 1
  max_datasets: 3                # Never load more than 3
```

#### Code Execution Security Configuration

```yaml
code_execution_security:
  enabled: true                  # Master kill switch

  # Code validation (AST-based)
  validation:
    use_ast_visitor: true        # Use AST validation (recommended)
    block_dunder_attributes: true # Block __globals__, __class__, etc.
    allowed_imports:             # Whitelist of safe imports
      - pandas
      - numpy
      - matplotlib

  # Sandbox execution
  isolation:
    use_process_isolation: false # Process isolation (disabled in Docker)
    cpu_time_limit_seconds: 5   # Max CPU time
    memory_limit_mb: 512         # Max memory usage
    wall_time_limit_seconds: 10 # Max wall clock time

  # Audit logging
  audit:
    enabled: true
    log_file: "logs/code_execution_audit.log"
    max_bytes: 104857600         # 100MB
    backup_count: 10
```

### Environment Variables (`backend/.env`)

```bash
# Required: LLM Provider API Keys
OPENAI_API_KEY=sk-...                     # Required for embeddings
ANTHROPIC_API_KEY=sk-ant-...              # Recommended for LLM
GOOGLE_API_KEY=...                        # Optional fallback

# Database
DATABASE_URL=postgresql+asyncpg://govtech:govtech_dev@postgres:5432/govtech_rag

# Optional: LangSmith Tracing
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=govtech-multi-agent-system

# Optional: Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1                                 # Uvicorn workers
```

---

<a name="api-documentation"></a>
## 📚 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Chat Endpoints

**POST /api/chat/**
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me employment in 2023",
    "conversation_id": null
  }'
```

Response:
```json
{
  "message": "Employment in 2023 totaled...",
  "visualization": {...},
  "sources": ["mrsd_2024.csv"],
  "agent_trace": [...],
  "conversation_id": "uuid"
}
```

**GET /api/chat/history/{conversation_id}**
```bash
curl http://localhost:8000/api/chat/history/{conversation_id}
```

**DELETE /api/chat/history/{conversation_id}**
```bash
curl -X DELETE http://localhost:8000/api/chat/history/{conversation_id}
```

#### Configuration Endpoints

**GET /api/config/health**
```bash
curl http://localhost:8000/api/config/health
```

Response:
```json
{
  "status": "healthy",
  "database": "connected",
  "vector_store": "ready"
}
```

**GET /api/config/providers**
```bash
curl http://localhost:8000/api/config/providers
```

#### Data Endpoints

**GET /api/data/datasets**
```bash
curl http://localhost:8000/api/data/datasets
```

**GET /api/data/datasets/{path}/info**
```bash
curl http://localhost:8000/api/data/datasets/employment_2023.csv/info
```

---

<a name="development"></a>
## 👨‍💻 Development

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (hot reload)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint

# Type check
npx tsc --noEmit

# Run tests
npm test                    # Run all tests
npm run test:watch          # Watch mode (auto-rerun on changes)
npm run test:coverage       # With coverage report
npm run test:ci             # CI mode (non-interactive)
```

### Backend Development

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For testing

# Format code
black app/
isort app/

# Lint code
flake8 app/
mypy app/

# Run backend
uvicorn app.main:app --reload
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Adding New Datasets

1. Place CSV/Excel files in `dataset/` directory
2. Run ingestion: `make ingest`
3. Verify ingestion: `make db-shell` and query metadata tables

### Code Style

- **Python**: Black + isort + flake8 + mypy
- **TypeScript**: ESLint + Prettier
- **Commit Messages**: Conventional Commits (feat, fix, docs, etc.)

---

<a name="troubleshooting"></a>
## 🐛 Troubleshooting

### Common Issues

#### 1. "Database connection failed"

**Symptom**: Backend logs show `asyncpg.exceptions.InvalidCatalogNameError`

**Solution**:
```bash
# Check if PostgreSQL is running
docker-compose ps

# Recreate database
make down
make clean
make up
```

#### 2. "OpenAI API key not found"

**Symptom**: `ValueError: OpenAI API key not found in environment`

**Solution**:
```bash
# Check environment file
cat backend/.env | grep OPENAI_API_KEY

# Ensure key is set
echo "OPENAI_API_KEY=sk-..." >> backend/.env

# Restart backend
docker-compose restart backend
```

#### 3. "No datasets found"

**Symptom**: Queries return "I couldn't find relevant datasets"

**Solution**:
```bash
# Run ingestion manually
make ingest

# Check if data was ingested
make db-shell
SELECT COUNT(*) FROM employment_dataset_metadata;
```

#### 4. "Frontend can't connect to backend"

**Symptom**: Network error in browser console

**Solution**:
```bash
# Check if backend is running
curl http://localhost:8000/api/config/health

# Check Docker networks
docker network ls
docker network inspect govtech_assignment_default

# Restart services
docker-compose restart
```

#### 5. "LLM request timeout"

**Symptom**: `ReadTimeout: Request timed out`

**Solution**:
- Check API key is valid
- Verify network connectivity
- Increase timeout in `config.yaml`:
  ```yaml
  llm:
    providers:
      anthropic:
        timeout: 60  # Increase from 30
  ```

#### 6. "Vector index not found"

**Symptom**: `InvalidParameterValue: ivfflat index not found`

**Solution**:
```bash
# Rebuild vector indexes
make db-shell

# Drop and recreate indexes
DROP INDEX IF EXISTS idx_employment_metadata_embedding;
CREATE INDEX idx_employment_metadata_embedding
ON employment_dataset_metadata
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Debug Mode

Enable verbose logging:

```bash
# Edit docker-compose.yml
environment:
  - LOG_LEVEL=DEBUG

# Restart
docker-compose restart backend

# View logs
docker-compose logs -f backend
```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-org/govtech-chat-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/govtech-chat-assistant/discussions)
- **Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Singapore Government Open Data**: For providing rich datasets
- **LangChain/LangGraph**: For agent orchestration framework
- **Anthropic**: For Claude models with excellent reasoning
- **OpenAI**: For embedding models and GPT
- **pgvector**: For native PostgreSQL vector search

---

**Built with ❤️ for Singapore Government Data Analysis**
