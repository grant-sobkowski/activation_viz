SHELL := /bin/bash
PACKAGE_SLUG=activation_viz
PYTHON_VERSION := $(shell cat .python-version)

ifeq ($(USE_SYSTEM_PYTHON), true)
	PYTHON_VENV :=
	UV := uv
else
	PYTHON_VENV := .venv
	UV := uv
endif

.PHONY: help
help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

.venv:
	$(UV) venv --python $(PYTHON_VERSION)

uv.lock: pyproject.toml
	$(UV) lock

.PHONY: sync
sync: $(PYTHON_VENV) uv.lock  ## Install/sync dependencies, create .venv
	@command -v uv >/dev/null 2>&1 || { echo >&2 "uv is not installed. Installing via pip..."; pip install uv; }
	$(UV) sync --group dev

.PHONY: pre-commit
pre-commit: sync  ## Install pre-commit hooks
	$(UV) run pre-commit install

.PHONY: test
test:  ## Run pytest with coverage
	$(UV) run pytest --cov=./${PACKAGE_SLUG} --cov-report=term-missing tests

.PHONY: lint
lint:  ## Check types (mypy) and lint/format (ruff)
	@status=0; \
	echo "==> mypy"; \
	$(UV) run mypy ${PACKAGE_SLUG} || status=1; \
	echo "==> ruff check"; \
	$(UV) run ruff check . || status=1; \
	echo "==> ruff format --check"; \
	$(UV) run ruff format . --check || status=1; \
	if [ $$status -ne 0 ]; then \
		echo ""; \
		echo "Lint failed. Run 'make lint-fix' to auto-fix formatting/lint issues."; \
	fi; \
	exit $$status

.PHONY: lint-fix
lint-fix:  ## Auto-fix lint and formatting issues with ruff
	$(UV) run ruff check . --fix
	$(UV) run ruff format .

.PHONY: build
build: sync  ## Build the package (wheel + sdist)
	$(UV) run python -m build
