provider "google" {
  project = var.project_id
  region  = var.default_region
}

provider "google-beta" {
  project = var.project_id
  region  = var.default_region
}

terraform {
  required_version = "~>1.14.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.40.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 7.26.0"
    }
  }

  backend "gcs" {
    bucket = "haru256-sandbox-20240502-tfstate"
  }
}
