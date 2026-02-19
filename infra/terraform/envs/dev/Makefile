.DEFAULT_GOAL := help
include ../../scripts/common.mk

.PHONY: plan
plan: # terraform plan
	terraform plan

.PHONY: destroy
destroy: # terraform destroy chargeable resources for gke
	terraform destroy \
		-target=module.gke.google_container_cluster.sandbox \
		-target=module.gke.google_compute_router_nat.gke \
		-target=module.gke.google_compute_router.gke

.PHONY: apply
apply: # terraform apply
	terraform apply

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done
