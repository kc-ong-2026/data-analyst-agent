# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

### Docker (Recommended)
```bash
make dev          # Development with hot reload (frontend:3000, backend:8000)
make up           # Production containers
make down         # Stop containers
make logs         # View all logs
make logs-backend # View backend logs only
make clean        # Remove containers and volumes
```

### Local Development
```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add API keys
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

### Frontend Commands
```bash
npm run dev       # Vite dev server (port 5173)
npm run build     # TypeScript check + Vite build
npm run lint      # ESLint with zero warnings tolerance
npm run preview   # Preview production build
```

## Architecture

### Multi-Agent LangGraph System
The backend implements a three-agent workflow using LangGraph, orchestrated by `AgentOrchestrator`:

1. **DataCoordinatorAgent** (`coordinator_agent.py`)
   - Entry point for all queries
   - Analyzes user intent, identifies required datasets, creates execution plan
   - Flow: `analyze_query -> identify_data_sources -> create_plan -> determine_delegation`

2. **DataExtractionAgent** (`extraction_agent.py`)
   - Matches queries to Singapore government datasets
   - Loads CSV/Excel files via `DataService`
   - Flow: `identify_datasets -> load_datasets -> extract_relevant_data -> format_output`

3. **AnalyticsAgent** (`analytics_agent.py`)
   - Generates insights and visualization specs for the frontend
   - Flow: `prepare_data -> analyze_data -> generate_visualization -> compose_response`

Agent communication uses shared `GraphState` (TypedDict) with fields: `messages`, `current_task`, `extracted_data`, `analysis_results`, `workflow_plan`, `errors`, `metadata`.

### LLM Provider Abstraction
`LLMService` (`llm_service.py`) provides model-agnostic LLM access:
- Supports OpenAI, Anthropic, Google via LangChain integrations
- Configuration in `backend/config/config.yaml`, API keys in `.env`
- Agents get LLM via `self.get_llm()` which calls `get_llm_service()`

### Frontend State Management
- Zustand store (`chatStore.ts`) manages chat state and visualization
- API client (`api/client.ts`) wraps axios for `/api/chat`, `/api/data`, `/api/config/health` endpoints
- Recharts renders visualizations based on `VisualizationData` specs from analytics agent
- **Security**: LLM provider/model selection is server-side only and not exposed to frontend

## Key Files

| Path | Purpose |
|------|---------|
| `backend/app/services/agents/orchestrator.py` | Top-level LangGraph that routes between agents |
| `backend/app/services/agents/base_agent.py` | `BaseAgent` ABC, `AgentState`, `GraphState` definitions |
| `backend/app/config.py` | `AppConfig` merges YAML config with env vars |
| `backend/app/services/data_service.py` | Loads/queries CSV and Excel datasets |
| `frontend/src/store/chatStore.ts` | Zustand store with `sendMessage`, `clearMessages` |
| `frontend/src/components/VisualizationPanel.tsx` | Recharts renderer for bar/line/pie/scatter charts |

## Configuration

### Environment Variables (`backend/.env`)
API keys only (for security):
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- `DATABASE_URL`

### LLM Configuration (`backend/config/config.yaml`)
All LLM and embedding configuration (server-side only, not exposed to frontend):
- `llm.default_provider` - Default LLM provider (openai/anthropic/google)
- `llm.providers.<provider>.default_model` - Default model per provider
- `llm.providers.<provider>.temperature` - Temperature per provider
- `llm.providers.<provider>.max_tokens` - Max tokens per provider
- `embeddings.default_provider` - Default embedding provider
- `embeddings.providers.<provider>.default_model` - Default embedding model per provider

**Note**: For security, configuration details are never exposed to the frontend. The backend always uses settings from `config.yaml`.

## Data Sources

- `dataset/singapore_manpower_dataset/` - Employment, income, labour force, hours worked (CSV/Excel)
- `api_spec/singapore_environment_api_specs/` - Weather, air quality, flood API specs (JSON)

Datasets are accessed via `DataService.get_available_datasets()` and loaded with pandas.

## Post-Change Validation: Linting & Tests

**IMPORTANT:** After making any code changes, Claude Code MUST fix all linting issues and ensure all tests pass. This validation happens automatically as part of the development workflow - commits are left for the user to create manually.

### Validation Workflow

After any changes:
1. ✅ **Fix linting** - Auto-fix with Black/Ruff (backend) and ESLint (frontend)
2. ✅ **Run tests** - Execute unit and integration tests to verify changes
3. ✅ **Report status** - Summarize what passed/failed
4. ⏸️ **Do NOT commit** - User handles git commits manually

### Required Checks

### Backend Checks

```bash
cd backend

# 1. Ruff linter (Python linting)
ruff check .

# 2. Black formatter (Python code formatting)
black --check .

# 3. Type checking (mypy - optional, continues on error in CI)
mypy app/ --ignore-missing-imports

# 4. Unit tests (CI/CD: backend-unit-tests job)
pytest tests/unit/ -v --tb=short --maxfail=5

# 5. Integration tests (CI/CD: backend-integration-tests job)
# IMPORTANT: Only run tests specified in CI/CD pipeline
# Start postgres first: docker-compose up -d postgres
pytest tests/integration/test_e2e_pipeline.py tests/integration/test_orchestrator.py -v --tb=short

# 6. Security tests (CI/CD: backend-security-tests job)
pytest tests/security/ -v --tb=short --maxfail=5
```

### Frontend Checks

```bash
cd frontend

# 1. ESLint (JavaScript/TypeScript linting)
npm run lint

# 2. TypeScript type check
npx tsc --noEmit

# 3. Jest tests with coverage (CI/CD: frontend-unit-tests job)
# IMPORTANT: Use CI test command to match CI/CD pipeline
npm run test:ci
```

### Quick Validation (All Checks)

Run everything at once:

```bash
# Backend (from project root)
cd backend && \
  ruff check . && \
  black --check . && \
  pytest tests/unit/ -v --tb=short --maxfail=5 && \
  pytest tests/integration/ -v --tb=short && \
  echo "✅ Backend checks passed!"

# Frontend (from project root)
cd frontend && \
  npm run lint && \
  npx tsc --noEmit && \
  npm test -- --passWithNoTests && \
  echo "✅ Frontend checks passed!"
```

### Docker-based Validation (Matches CI Exactly)

Use Docker to run checks in the same environment as CI:

```bash
# Backend lint and tests (matches CI/CD exactly)
docker-compose -f docker-compose.yml run --rm backend bash -c "
  ruff check . && \
  black --check . && \
  pytest tests/unit/ -v --tb=short && \
  pytest tests/integration/test_e2e_pipeline.py tests/integration/test_orchestrator.py -v --tb=short && \
  pytest tests/security/ -v --tb=short
"

# Frontend lint and tests
docker-compose -f docker-compose.yml run --rm frontend bash -c "
  npm run lint && \
  npx tsc --noEmit && \
  npm test -- --passWithNoTests
"
```

### Auto-fix Common Issues

```bash
# Backend: Auto-format with Black
cd backend && black .

# Backend: Auto-fix Ruff issues
cd backend && ruff check . --fix

# Frontend: Auto-fix ESLint issues
cd frontend && npm run lint -- --fix
```

### Success Criteria (After Changes)

Claude Code ensures these all pass after any modifications (matching CI/CD pipeline):
- ✅ `ruff check .` → All checks passed!
- ✅ `black --check .` → All done! X files would be left unchanged
- ✅ `npm run lint` → No output (zero warnings, zero errors)
- ✅ `npx tsc --noEmit` → No output (zero errors)
- ✅ Unit tests pass → `pytest tests/unit/` succeeds
- ✅ Integration tests pass → `pytest tests/integration/test_e2e_pipeline.py tests/integration/test_orchestrator.py` succeeds
- ✅ Security tests pass → `pytest tests/security/` succeeds
- ✅ Frontend tests pass → `npm run test:ci` succeeds

**Note:** Git commits remain manual - Claude Code only validates code quality.

### CI/CD Pipeline Jobs

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs these jobs:
1. **backend-lint** - Ruff + Black + mypy
2. **backend-unit-tests** - Unit tests with coverage
3. **backend-integration-tests** - Integration tests (orchestrator, e2e)
4. **backend-security-tests** - Security validation tests
5. **frontend-lint** - ESLint + TypeScript check
6. **frontend-unit-tests** - Jest tests with coverage
7. **docker-build** - Build both Docker images

All jobs must pass for `ci-success` to complete.
