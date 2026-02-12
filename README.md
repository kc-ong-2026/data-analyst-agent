# Singapore Government Data Chat Assistant

An intelligent AI-powered chat assistant that enables natural language querying and analysis of Singapore government datasets with real-time visualization capabilities. Built with a multi-agent architecture using LangGraph, hybrid RAG retrieval, and resilient LLM infrastructure.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Technology Stack & Justifications](#-technology-stack--justifications)
- [Setup Instructions](#-setup-instructions)
- [Running the Application](#-running-the-application)
- [Running Tests](#-running-tests)
- [Sample Queries](#-sample-queries)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

This system allows users to query Singapore government datasets (employment, income, hours worked) using natural language and receive intelligent insights with visualizations. It uses a sophisticated multi-agent pipeline that validates queries, retrieves relevant data using hybrid RAG (vector + BM25 search), generates pandas code for analysis, and creates visualizations.

### What Makes This System Unique?

- **Multi-Agent Intelligence**: Four specialized agents (Verification, Coordinator, Extraction, Analytics) work together to handle complex queries
- **Hybrid RAG Retrieval**: Combines vector search, BM25 keyword matching, RRF fusion, and cross-encoder reranking for optimal dataset selection
- **LLM Resilience**: Automatic retry, model fallback, and provider fallback with circuit breaker pattern ensures reliability
- **Safe Code Generation**: Generates pandas code (not SQL) and executes in a sandboxed environment
- **Confidence-Based Loading**: Intelligently loads 1-3 most relevant datasets based on reranker scores
- **Real-Time Streaming**: Progressive response delivery via Server-Sent Events (SSE)

---

## ✨ Key Features

### Core Capabilities
- 🤖 **Natural Language Queries**: Ask questions in plain English (e.g., "Show me employment trends in 2023 by age")
- 📊 **Dynamic Visualizations**: Auto-generated bar, line, pie, and scatter charts
- 🔍 **Intelligent Data Retrieval**: Hybrid search finds the most relevant datasets
- 💡 **Contextual Analysis**: Multi-agent system understands query intent and generates insights
- 📈 **Real-Time Streaming**: See responses as they're generated

### Technical Features
- 🏗️ **Multi-Agent Architecture**: LangGraph-based agent orchestration
- 🔄 **LLM Provider Agnostic**: Supports OpenAI, Anthropic, and Google models
- 🛡️ **Resilient Infrastructure**: Automatic retry and fallback mechanisms
- 🐳 **Docker Ready**: Full containerization for easy deployment
- 🧪 **Comprehensive Testing**: Unit, integration, and evaluation tests
- 📝 **Type Safety**: Full TypeScript frontend and Pydantic backend

---

## 🏛️ Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (React + TypeScript)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Chat UI      │  │ Visualization│  │ Zustand      │          │
│  │ (SSE Stream) │  │ Panel        │  │ Store        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP/SSE
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI + LangGraph)                 │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Agent Orchestrator (LangGraph)                 │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │ │
│  │  │Verifica- │→ │Coordina- │→ │Extrac-   │→ │Analytics │  │ │
│  │  │tion      │  │tor       │  │tion      │  │          │  │ │
│  │  │(Validate)│  │(Plan)    │  │(RAG)     │  │(Analyze) │  │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│           │                 │                 │                  │
│  ┌────────▼─────┐  ┌───────▼──────┐  ┌──────▼───────┐         │
│  │ RAG Service  │  │ LLM Resilience│  │ Data Service │         │
│  │ (Vector+BM25)│  │ (Retry+       │  │ (DataFrame   │         │
│  │  +Reranking  │  │  Fallback)    │  │  Operations) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         PostgreSQL with pgvector Extension                       │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ employment_metadata  │  │ income_metadata      │            │
│  │ (embeddings + BM25)  │  │ (embeddings + BM25)  │            │
│  └──────────────────────┘  └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Agent Workflow

```
User Query → Verification Agent → [Valid?] → Coordinator Agent
                    ↓ Invalid              ↓
                   END                Data Extraction Agent
                                           ↓
                                    Analytics Agent
                                           ↓
                              Response + Visualization
```

**For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md)**

---

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

## 🧪 Running Tests

The project includes comprehensive testing with unit tests, integration tests, and evaluation tests.

### Quick Test Commands

```bash
# Run all tests
make test

# Run specific test categories
make test-unit              # Unit tests only (fast, ~30s)
make test-integration       # Integration tests with mocked LLM (fast)
make test-e2e              # End-to-end pipeline tests

# Run with coverage report
make test-coverage
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

### Test Structure

```
backend/tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_verification_agent.py
│   ├── test_coordinator_agent.py
│   ├── test_extraction_agent.py
│   ├── test_analytics_agent.py
│   └── test_rag_retrieval.py
│
├── integration/             # Integration tests (slower, real flows)
│   ├── test_e2e_pipeline.py
│   ├── test_orchestrator.py
│   └── test_rag_pipeline.py
│
├── evaluation/              # Quality metrics
│   ├── test_retrieval_metrics.py
│   └── test_generation_metrics.py
│
└── test_smoke.py           # Quick validation
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

---

## 💬 Sample Queries

Here are example queries you can try in the chat interface to demonstrate the system's capabilities:

### Basic Employment Queries

```
1. "Show me employment data for 2023"
   - Retrieves employment statistics for 2023
   - Creates visualization of employment by category

2. "What is the unemployment rate in 2022?"
   - Calculates unemployment rate from employment data
   - Provides explanation and context

3. "Compare employment between 2020 and 2023"
   - Multi-year comparison
   - Line chart showing trends over time
```

### Demographic Analysis

```
4. "Show me employment by age group in 2023"
   - Breaks down employment by age categories
   - Bar chart with age distribution

5. "What is the employment rate for males vs females in 2023?"
   - Gender-based comparison
   - Grouped bar chart showing differences

6. "Employment trends for people aged 55 and above from 2019 to 2023"
   - Time series for specific age group
   - Line chart showing elderly employment trends
```

### Industry Analysis

```
7. "Which industries have the highest employment in 2023?"
   - Ranks industries by employment numbers
   - Bar chart with top industries

8. "Show me employment in the technology sector over the last 5 years"
   - Industry-specific trend analysis
   - Line chart with tech employment growth

9. "Compare employment in manufacturing vs services sector"
   - Cross-industry comparison
   - Grouped bar chart
```

### Income Analysis

```
10. "What is the average income in 2023?"
    - Calculates mean income statistics
    - Provides context about income distribution

11. "Show me income trends by qualification level"
    - Income analysis by education
    - Bar chart comparing qualification levels

12. "Compare income between different age groups in 2023"
    - Age-based income analysis
    - Bar chart showing income by age
```

### Hours Worked Analysis

```
13. "What are the average working hours in 2023?"
    - Calculates mean hours worked
    - Provides comparison to previous years

14. "Show me working hours by industry"
    - Industry-specific hours analysis
    - Bar chart with hours by sector

15. "Compare working hours between full-time and part-time workers"
    - Employment type comparison
    - Grouped bar chart
```

### Complex Multi-Dimensional Queries

```
16. "Show me employment trends for females in the finance sector from 2020 to 2023"
    - Multi-filter query (gender + industry + time)
    - Line chart with specific segment

17. "What is the employment rate for degree holders aged 25-34 in 2023?"
    - Triple filter (education + age + year)
    - Provides specific statistics

18. "Compare income between males and females in the technology sector"
    - Multi-dimensional comparison (gender + industry + income)
    - Grouped bar chart
```

### Invalid Queries (for testing validation)

```
19. "What's the weather in Singapore?"
    ❌ Response: "I can only answer questions about Singapore employment, income, and hours worked data."

20. "Show me employment data"
    ❌ Response: "Please specify which year or year range you're interested in."

21. "How many people work in 2025?"
    ⚠️ Response: "Data for 2025 is not available. Available years: 2010-2023."
```

### Expected Response Format

For each valid query, you'll receive:

1. **Natural Language Explanation**: Human-readable insights
2. **Visualization** (if appropriate): Chart spec rendered by Recharts
3. **Sources**: List of datasets used
4. **Agent Trace**: Shows which agents processed the query

Example response for "Show me employment by age in 2023":

```json
{
  "message": "Employment in 2023 totaled 3.2 million. The 25-34 age group has the highest employment at 980,000, followed by 35-44 at 820,000...",
  "visualization": {
    "type": "bar",
    "title": "Employment by Age Group (2023)",
    "data": {...}
  },
  "sources": ["mrsd_2024_employment.csv"],
  "agent_trace": ["Query Verification", "Data Coordinator", "Data Extraction", "Analytics"]
}
```

---

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
npm run type-check
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
