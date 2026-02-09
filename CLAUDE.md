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
- Config in `backend/config/config.yaml` and environment variables
- Agents get LLM via `self.get_llm()` which calls `get_llm_service()`

### Frontend State Management
- Zustand store (`chatStore.ts`) manages chat state, visualization, provider selection
- API client (`api/client.ts`) wraps axios for `/api/chat`, `/api/config`, `/api/data` endpoints
- Recharts renders visualizations based on `VisualizationData` specs from analytics agent

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

Environment variables in `backend/.env`:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- `DEFAULT_LLM_PROVIDER` (openai/anthropic/google)
- `DEFAULT_LLM_MODEL`

YAML config in `backend/config/config.yaml` defines per-provider model lists, temperatures, max_tokens.

## Data Sources

- `dataset/singapore_manpower_dataset/` - Employment, income, labour force, hours worked (CSV/Excel)
- `api_spec/singapore_environment_api_specs/` - Weather, air quality, flood API specs (JSON)

Datasets are accessed via `DataService.get_available_datasets()` and loaded with pandas.
