.DEFAULT_GOAL := help
HAS_CUDA := $(shell command -v nvcc 2> /dev/null && echo 1 || echo 0)

.PHONY: lint
lint: ## Run Linter
	uv run ruff check .
	uv run mypy .

.PHONY: fmt
fmt: ## Run formatter
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: test
test: ## Run tests
	uv run pytest . -v -s

.PHONY: lock
lock: ## Lock dependencies
	uv lock

.PHONY: install
install: ## Setup the project
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		echo "=== Install with GPU support ==="; \
		uv sync --all-groups --extra=gpu; \
	else \
		echo "=== Install with CPU support ==="; \
		uv sync --all-groups --extra=cpu; \
	fi

.PHONY: clean
clean: ## Remove caches
	rm -rf .mypy_cache .pytest_cache
	fd -H --type directory __pycache__ . -x rm -rf

.PHONY: help
help: ## Show options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
