.PHONY: lint fmt help test lock setup train
.DEFAULT_GOAL := help

lint: ## Run Linter
	uv run ruff check .
	uv run mypy .

fmt: ## Run formatter
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run tests
	uv run pytest .

lock: ## Lock dependencies
	uv lock

install: ## Setup the project
	uv sync --all-groups

train: ## Train the model
	uv run python train.py

help: ## Show options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
