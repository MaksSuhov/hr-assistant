PYTHON ?= python

.PHONY: help install lint format format-check typecheck test check web-build

help:
	@echo "install       Install the API with dev dependencies"
	@echo "lint          Run ruff lint"
	@echo "format        Apply ruff formatting"
	@echo "format-check  Verify ruff formatting without writing"
	@echo "typecheck     Run mypy over src and eval"
	@echo "test          Run pytest with coverage"
	@echo "check         Run lint, format-check, typecheck and tests"
	@echo "web-build     Typecheck and build the React frontend"

install:
	$(PYTHON) -m pip install -e "apps/api[dev]"

lint:
	cd apps/api && ruff check .

format:
	cd apps/api && ruff format .

format-check:
	cd apps/api && ruff format --check .

typecheck:
	cd apps/api && mypy

test:
	cd apps/api && pytest --cov=hr_assistant --cov=eval --cov-report=term-missing

check: lint format-check typecheck test

web-build:
	npm ci
	npm run build