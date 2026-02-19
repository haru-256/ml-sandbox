# Training Job実行用service account
resource "google_service_account" "training_sa" {
  account_id   = "custom-training-job"
  display_name = "Vertex AI Custom Training SA"
}

# リソースへのアクセス権限付与
resource "google_project_iam_member" "training_sa_roles" {
  for_each = toset([
    "roles/artifactregistry.reader", # イメージPull
    "roles/logging.logWriter",       # ログ書き込み
    "roles/storage.objectAdmin",     # GCS読み書き
    "roles/aiplatform.user",         #  Vertex SDKで実験管理するため
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.training_sa.email}"
}

# ジョブ実行者への SA 利用権限付与
resource "google_service_account_iam_member" "user_act_as_sa" {
  for_each           = toset(var.users)
  service_account_id = google_service_account.training_sa.name
  role               = "roles/iam.serviceAccountUser"
  member             = "user:${each.value}"
}

# artifact registory
resource "google_artifact_registry_repository" "ml_sandbox" {
  location      = var.region
  repository_id = "ml-sandbox"
  description   = "ML Sandbox Image Registry"
  format        = "DOCKER"
  vulnerability_scanning_config {
    enablement_config = "DISABLED"
  }
}
