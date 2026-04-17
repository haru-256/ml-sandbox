.DEFAULT_GOAL := help

.PHONY: help

help: ## Show options
	@awk 'BEGIN { FS=":.*?## " } /^[a-zA-Z_-]+:.*?## / { descriptions[$$1] = $$2; order[++count] = $$1 } END { for (i = 1; i <= count; i++) if (!(order[i] in first_seen)) first_seen[order[i]] = i; for (i = 1; i <= count; i++) { target = order[i]; if (first_seen[target] == i) printf "\033[36m%-20s\033[0m %s\n", target, descriptions[target] } }' $(MAKEFILE_LIST)
