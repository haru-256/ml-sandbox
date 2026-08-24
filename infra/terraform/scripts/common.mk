TFLINT_CONFIG := $(shell pwd)/../../.tflint.hcl

.PHONY: format
format: # format terraform
	terraform fmt -recursive

.PHONY: format-check
format-check: # check terraform formatting
	terraform fmt -check -recursive

.PHONY: lint
lint: # lint terraform
	tflint --init --config $(TFLINT_CONFIG)
	tflint --recursive --config $(TFLINT_CONFIG)
	trivy config . --severity=HIGH,CRITICAL

.PHONY: validate
validate: # validate terraform without remote backend
	terraform init -backend=false
	terraform validate
