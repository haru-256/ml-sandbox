.PHONY: lint fmt help
.DEFAULT_GOAL := help

lint: ## Run Linter
	uvx ruff check .
	uv run mypy .

fmt: ## Run formatter
	uvx ruff check --fix .
	uvx ruff format .

test: ## Run tests
	uv run pytest .

lock: ## Lock dependencies
	uv lock

setup: ## Setup the project
	uv sync
	uv tool install ruff@0.6.3

help: ## Show options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
