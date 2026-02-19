variable "project_id" {
  type        = string
  description = "The ID for your GCP project"
}

variable "required_services" {
  type        = list(string)
  description = "The service names enabled"
}

variable "wait_seconds" {
  type        = number
  description = "seconds to wait"
}
