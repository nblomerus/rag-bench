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

help:
	@echo "  pyenv          Create Python virtualenv with pyenv
	@echo "  upgrade        Compile requirements
	@echo "  install        Install/sync dependencies
	@echo "  check          Run pre-commit checks
	@echo "  ruff           Run ruff formatter and linter
	@echo "  test           Run tests
	@echo "  clean          Remove Python artifacts

.PHONY: pyenv upgrade install pre-commit check ruff test clean help
