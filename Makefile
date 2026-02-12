.PHONY: help build up down dev clean logs ingest db-shell test test-smoke test-e2e test-unit test-integration test-security test-evaluation test-evaluation-retrieval test-evaluation-generation test-evaluation-export test-coverage test-frontend test-frontend-watch test-frontend-coverage test-all

help:
	@echo "Available commands:"
	@echo "  make build             - Build Docker images"
	@echo "  make up                - Start production containers"
	@echo "  make down              - Stop containers"
	@echo "  make dev               - Start development containers with hot reload"
	@echo "  make clean             - Remove containers and volumes"
	@echo "  make logs              - View container logs"
	@echo "  make ingest            - Run data ingestion pipeline"
	@echo "  make db-shell          - Open psql shell to database"
	@echo "  make test              - Run all tests in Docker container"
	@echo "  make test-smoke        - Run smoke tests (quick validation)"
	@echo "  make test-e2e          - Run end-to-end pipeline tests"
	@echo "  make test-unit         - Run unit tests only (optional: TEST=file.py or TEST=file.py::test_name)"
	@echo "  make test-integration         - Run FAST mocked tests (no LLM, no DB - 10s)"
	@echo "  make test-integration-with-db - Run integration tests with DB (skip LLM - slower)"
	@echo "  make test-integration-full    - Run ALL integration tests including LLM (VERY SLOW)"
	@echo "  make test-security            - Run security tests (code validation, sandboxing, audit)"
	@echo "  make test-evaluation          - Run all evaluation tests"
	@echo "  make test-evaluation-retrieval - Run retrieval metrics tests only"
	@echo "  make test-evaluation-generation - Run generation metrics tests only"
	@echo "  make test-evaluation-export   - Run evaluation and export for fine-tuning"
	@echo "  make test-coverage            - Run backend tests with coverage report"
	@echo ""
	@echo "Frontend Tests:"
	@echo "  make test-frontend            - Run frontend Jest tests in Docker"
	@echo "  make test-frontend-watch      - Run frontend tests in watch mode"
	@echo "  make test-frontend-coverage   - Run frontend tests with coverage report"
	@echo "  make test-all                 - Run ALL tests (backend + frontend)"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

dev:
	docker-compose -f docker-compose.dev.yml up --build

clean:
	docker-compose down -v --rmi local
	docker-compose -f docker-compose.dev.yml down -v --rmi local

logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

ingest:
	docker-compose exec backend python -m app.services.ingestion.ingest_cli

db-shell:
	docker-compose exec postgres psql -U govtech -d govtech_rag

test:
	@echo "Running all tests in Docker container..."
	docker-compose exec backend pytest tests/ -v --tb=short

test-smoke:
	@echo "Running smoke tests (quick validation)..."
	docker-compose exec backend pytest tests/test_smoke.py -v

test-e2e:
	@echo "Running end-to-end pipeline tests..."
	docker-compose exec backend pytest tests/integration/test_e2e_pipeline.py -v

test-unit:
	@echo "Running unit tests in Docker container..."
	docker-compose exec backend pytest tests/unit/$(if $(TEST),$(TEST),) -v --tb=short

test-integration:
	@echo "Running fast integration tests (mocked, no LLM, no DB)..."
	docker-compose exec backend pytest tests/integration/test_e2e_pipeline.py tests/integration/test_orchestrator.py -v --tb=short

test-integration-with-db:
	@echo "Running integration tests with database (skipping LLM tests)..."
	docker-compose exec backend pytest tests/integration/ -v --tb=short -m "not requires_llm"

test-integration-full:
	@echo "Running ALL integration tests (including LLM tests - SLOW)..."
	docker-compose exec backend pytest tests/integration/ -v --tb=short

test-security:
	@echo "Running security tests (code validation, sandboxing, audit logging)..."
	docker-compose exec backend pytest tests/security/ -v --tb=short

test-evaluation:
	@echo "Running all evaluation tests in Docker container..."
	docker-compose exec backend pytest tests/evaluation/ -v --tb=short

test-evaluation-retrieval:
	@echo "Running retrieval metrics evaluation tests..."
	docker-compose exec backend pytest tests/evaluation/test_retrieval_metrics.py -v -s

test-evaluation-generation:
	@echo "Running generation metrics evaluation tests..."
	docker-compose exec backend pytest tests/evaluation/test_generation_metrics.py -v -s

test-evaluation-export:
	@echo "Running evaluation tests and exporting results for fine-tuning..."
	docker-compose exec backend python scripts/run_evaluation.py --export-format openai --output-dir ./evaluation_results --generate-mock

test-coverage:
	@echo "Running backend tests with coverage report..."
	docker-compose exec backend pytest tests/ --cov=app --cov-report=html --cov-report=term -v

# Frontend Tests
test-frontend:
	@echo "Running frontend Jest tests in Docker container..."
	docker-compose exec frontend npm test

test-frontend-watch:
	@echo "Running frontend tests in watch mode (local)..."
	cd frontend && npm run test:watch

test-frontend-coverage:
	@echo "Running frontend tests with coverage report..."
	docker-compose exec frontend npm run test:coverage

# Run all tests (backend + frontend)
test-all:
	@echo "Running ALL tests (backend + frontend)..."
	@echo "\n=== Backend Tests ==="
	docker-compose exec backend pytest tests/unit/ tests/security/ -v --tb=short
	@echo "\n=== Frontend Tests ==="
	docker-compose exec frontend npm test
