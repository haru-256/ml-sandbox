.PHONY: format lint help

.DEFAULT_GOAL := help

format: # format terraform
	terraform fmt -recursive

lint: # lint terraform
	tflint --recursive --config $(shell pwd)/.tflint.hcl
	trivy config . --severity=HIGH,CRITICAL

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done
