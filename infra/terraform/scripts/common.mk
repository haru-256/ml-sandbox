.PHONY: format
format: # format terraform
	terraform fmt -recursive

.PHONY: lint
lint: # lint terraform
	tflint --recursive --config $(shell pwd)/.tflint.hcl
	trivy config . --severity=HIGH,CRITICAL
