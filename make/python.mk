HAS_CUDA := $(shell command -v nvcc 2> /dev/null && echo 1 || echo 0)

.PHONY: lint fmt test clean-cache lock

lint: ## Run linter
	uv run ruff check .
	uv run mypy .

fmt: ## Run formatter
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run tests
	uv run pytest . -v -s

clean-cache: ## Remove cache files
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	fd -H --type directory __pycache__ . -x rm -rf

lock: ## Lock dependencies
	uv lock
