.DEFAULT_GOAL := help

.PHONY: help
help: ## Show options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build the Docker
	docker image build -t us-docker.pkg.dev/haru256-ml-sandbox/ml-sandbox/job_runner:latest .

.PHONY: push
push: ## Push the Docker
	docker push us-docker.pkg.dev/haru256-ml-sandbox/ml-sandbox/job_runner:latest
