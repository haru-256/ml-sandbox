include ../../make/help.mk
include ../../make/python.mk

.PHONY: install
install: ## Setup the project
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		echo "=== Install with GPU support ==="; \
		uv sync --all-groups --extra=gpu; \
	else \
		echo "=== Install with CPU support ==="; \
		uv sync --all-groups --extra=cpu; \
	fi

# ─── Docker ───────────────────────────────────────────
DOCKER_IMAGE ?= recsys-candidate-generation:latest
# Additional build context for monorepo local path dep (required by Dockerfile)
DOCKER_BUILD_CONTEXTS := ml_sandbox_libs=../../libs/ml_sandbox_libs

.PHONY: docker-build
docker-build: ## Build the Docker image for local training
	docker build --platform linux/amd64 --build-context $(DOCKER_BUILD_CONTEXTS) -t $(DOCKER_IMAGE) .

.PHONY: docker-run
docker-run: ## Run training directly in Docker (requires WANDB_API_KEY; Hydra args via DOCKER_ARGS)
	@test -n "$$WANDB_API_KEY" || (echo "ERROR: WANDB_API_KEY is not set" && exit 1)
	docker run --rm --gpus all --platform linux/amd64 \
		-e WANDB_API_KEY=$$WANDB_API_KEY \
		-e EXPERIMENT_NAME=$${EXPERIMENT_NAME:-local-experiment} \
		$(DOCKER_IMAGE) uv run fit.py $(DOCKER_ARGS)

.PHONY: docker-up
docker-up: ## Start a dev shell container (use 'make docker-exec' to run training inside)
	docker compose up -d

.PHONY: docker-exec
docker-exec: ## Run training inside the dev shell container
	docker compose exec job uv run fit.py

.PHONY: docker-down
docker-down: ## Stop the dev shell container
	docker compose down
