# RAG-Bench Makefile

PYTHON_VERSION = 3.11
VIRTUALENV_NAME := $(shell basename $(CURDIR))

# Python environment

pyenv:
	pyenv install -s $(PYTHON_VERSION)
	pyenv virtualenv $(PYTHON_VERSION) $(VIRTUALENV_NAME)
	pyenv local $(VIRTUALENV_NAME)
	python -m pip install --upgrade pip

upgrade:
	uv pip compile -U requirements.in -o requirements.txt
	uv pip compile -U dev-requirements.in -o dev-requirements.txt

install:
	pip install --upgrade pip
	pip install uv
	uv pip sync requirements.txt dev-requirements.txt

# Code quality

pre-commit:
	pre-commit install

check: | pre-commit
	pre-commit run --all-files

ruff:
	ruff check --select I --fix
	ruff format

test:
	pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Deploy

run-server:
	python -m rag_bench.api.server

# Docker (Development) - Uses Docker Compose v2 (docker compose)

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-ps:
	docker compose ps

docker-exec-api:
	docker compose exec api /bin/bash

docker-clean:
	docker compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	docker system prune -f

docker-restart:
	docker compose restart

# Deployment

deploy-dev:
	@echo "Setting up RAG-Bench development environment..."
	chmod +x scripts/deploy-dev.sh
	./scripts/deploy-dev.sh

deploy:
	@echo "Starting RAG-Bench deployment to ragbench.co.za..."
	chmod +x scripts/deploy.sh scripts/init-ssl.sh
	./scripts/deploy.sh

# Environment Switching (local only)

switch-dev:
	@echo "Switching to development environment..."
	chmod +x scripts/toggle-env.sh
	./scripts/toggle-env.sh dev

switch-prod:
	@echo "Switching to production environment (local test)..."
	chmod +x scripts/toggle-env.sh
	./scripts/toggle-env.sh prod

env-status:
	@chmod +x scripts/toggle-env.sh
	@./scripts/toggle-env.sh status

deploy-ssl:
	@echo "Initializing Let's Encrypt SSL certificates..."
	chmod +x scripts/init-ssl.sh
	./scripts/init-ssl.sh

# Operations

backup:
	@echo "Creating backup..."
	chmod +x scripts/backup.sh
	./scripts/backup.sh

restore:
	@echo "Available backups:"
	@chmod +x scripts/restore.sh
	@./scripts/restore.sh || true

rollback:
	@echo "Available rollback points:"
	@chmod +x scripts/rollback.sh
	@./scripts/rollback.sh || true

health-monitor:
	@echo "Starting health monitor..."
	chmod +x scripts/health-monitor.sh
	./scripts/health-monitor.sh

# Production Operations (Single Machine - Downtime OK)

prod-start:
	@chmod +x scripts/prod-ops.sh
	@./scripts/prod-ops.sh start

prod-stop:
	@chmod +x scripts/prod-ops.sh
	@./scripts/prod-ops.sh stop

prod-restart:
	@chmod +x scripts/prod-ops.sh
	@./scripts/prod-ops.sh restart

prod-rebuild:
	@chmod +x scripts/prod-ops.sh
	@./scripts/prod-ops.sh rebuild

prod-ingest:
	@chmod +x scripts/prod-ops.sh
	@./scripts/prod-ops.sh ingest

prod-logs:
	@chmod +x scripts/prod-ops.sh
	@./scripts/prod-ops.sh logs

prod-status:
	@chmod +x scripts/prod-ops.sh
	@./scripts/prod-ops.sh status

prod-shell:
	@chmod +x scripts/prod-ops.sh
	@./scripts/prod-ops.sh shell

# Git 

prune-branches:
	git fetch --all --prune
	git fetch -p && for branch in $(git branch -vv | grep ': gone]' | awk '{print $1}'); do git branch -D $branch; done

help:
	@echo "RAG-Bench Makefile - Available commands:"
	@echo ""
	@echo "Python Environment:"
	@echo "  pyenv           Create Python virtualenv with pyenv"
	@echo "  upgrade         Compile requirements"
	@echo "  install         Install/sync dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  check           Run pre-commit checks"
	@echo "  ruff            Run ruff formatter and linter"
	@echo "  test            Run tests"
	@echo "  clean           Remove Python artifacts"
	@echo ""
	@echo "Development Server:"
	@echo "  run-server      Run API server locally"
	@echo ""
	@echo "Docker (Development):"
	@echo "  docker-build    Build Docker images"
	@echo "  docker-up       Start containers"
	@echo "  docker-down     Stop containers"
	@echo "  docker-logs     View container logs"
	@echo "  docker-ps       List running containers"
	@echo "  docker-restart  Restart all dev services"
	@echo "  docker-clean    Remove all containers and volumes"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-dev      Quick setup for local development"
	@echo "  deploy          Full deployment to ragbench.co.za"
	@echo "  deploy-ssl      Initialize Let's Encrypt SSL"
	@echo ""
	@echo "Environment Switching (local testing):"
	@echo "  switch-dev      Switch to development environment"
	@echo "  switch-prod     Switch to production environment (local)"
	@echo "  env-status      Show current environment"
	@echo ""
	@echo "Operations:"
	@echo "  backup          Create manual backup"
	@echo "  restore         Restore from backup"
	@echo "  rollback        Rollback to previous version"
	@echo "  health-monitor  Start health monitoring"
	@echo ""
	@echo "Production (Single Machine - Downtime OK):"
	@echo "  prod-start      Start production"
	@echo "  prod-stop       Stop production (free resources)"
	@echo "  prod-restart    Restart production"
	@echo "  prod-rebuild    Rebuild and restart production"
	@echo "  prod-ingest     Ingest papers into production"
	@echo "  prod-logs       View production logs"
	@echo "  prod-status     Check production status"
	@echo "  prod-shell      Shell into production container"
	@echo ""
	@echo "Git:"
	@echo "  prune-branches  Delete local branches removed from remote"
	@echo ""

.PHONY: pyenv upgrade install pre-commit check ruff test clean help prune-branches run-server \
        docker-build docker-up docker-down docker-logs docker-ps docker-exec-api docker-clean docker-restart \
        deploy-dev deploy deploy-ssl switch-dev switch-prod env-status backup restore rollback health-monitor \
        prod-start prod-stop prod-restart prod-rebuild prod-ingest prod-logs prod-status prod-shell
