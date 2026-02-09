.PHONY: help build up down dev clean logs ingest db-shell

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start production containers"
	@echo "  make down     - Stop containers"
	@echo "  make dev      - Start development containers with hot reload"
	@echo "  make clean    - Remove containers and volumes"
	@echo "  make logs     - View container logs"
	@echo "  make ingest   - Run data ingestion pipeline"
	@echo "  make db-shell - Open psql shell to database"

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
