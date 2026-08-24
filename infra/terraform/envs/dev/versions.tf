terraform {
  required_version = "~> 1.15.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.43.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 7.43.0"
    }
  }

  backend "gcs" {
    bucket = "haru256-sandbox-20240502-tfstate"
  }
}
