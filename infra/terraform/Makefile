.PHONY: format format-check lint validate help

.DEFAULT_GOAL := help

TFLINT_CONFIG := $(shell pwd)/.tflint.hcl

format: # format terraform
	terraform fmt -recursive

format-check: # check terraform formatting
	terraform fmt -check -recursive

lint: # lint terraform
	tflint --init --config $(TFLINT_CONFIG)
	tflint --recursive --config $(TFLINT_CONFIG)
	trivy config . --severity=HIGH,CRITICAL

validate: # validate terraform without remote backend
	$(MAKE) -C envs/dev validate

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done
