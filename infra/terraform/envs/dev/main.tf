locals {
  # このTerraform構成で必要な全APIをリスト化
  required_services = [
    "storage.googleapis.com", # GCSモジュール用
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "aiplatform.googleapis.com",
    "logging.googleapis.com",
    "compute.googleapis.com",
  ]
}

# google cloud project
data "google_project" "project" {
  project_id = var.project_id
}

# 必要なAPIをすべて有効化し待機
module "required_project_services" {
  source = "../../modules/google_project_services"

  project_id        = var.project_id
  required_services = local.required_services
  wait_seconds      = 60
}

# create the bucket for terraform state
module "tfstate_bucket" {
  source         = "../../modules/tfstate_gcs_bucket"
  gcp_project_id = data.google_project.project.project_id

  depends_on = [module.required_project_services]
}

# prepare vertex ai custom training jobs
module "vertex_ai_training" {
  source = "../../modules/vertex_ai_training"

  project_id = var.project_id
  region     = var.default_region
  users      = [var.owner_email]

  depends_on = [module.required_project_services]
}
