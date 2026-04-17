include ../../make/help.mk
include ../../make/python.mk

.PHONY: install train

install: ## Setup the project
	uv sync --all-groups

train: ## Train the model
	uv run python train.py
