terraform {
  required_version = ">= 1.14.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">=6.22.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">=6.22.0"
    }
    time = {
      source  = "hashicorp/time"
      version = ">= 0.9.0"
    }
  }
}
