variable "project_id" {
  type        = string
  description = "The ID of GCP project"
}

variable "region" {
  type        = string
  description = "The GCP region where resources will be created"
}

variable "users" {
  type        = list(string)
  description = "The users to be granted access to the service account"
}
