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
	uv run pytest .

.PHONY: lock
lock: pyproject.toml # lock uv
	uv lock

.PHONY: install
install: uv.lock # install package by uv
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		echo "=== Install with GPU support ==="; \
		uv sync --all-groups --extra=gpu; \
	else \
		echo "=== Install with CPU support ==="; \
		uv sync --all-groups --extra=cpu; \
	fi

.PHONY: train
train: ## Train the model
	uv run python train.py

.PHONY: help
help: ## Show options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
