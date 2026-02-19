# 必要なAPIをすべて有効化するリソース
resource "google_project_service" "api_services" {
  # リストをセットに変換して for_each でループ
  for_each = toset(var.required_services)

  project = var.project_id
  service = each.value

  # terraform destroy 時にサービスを無効化しない
  disable_on_destroy = false
}

// GCE APIが有効化されるまで待機するリソース
resource "time_sleep" "wait_for_gce_api" {
  for_each        = google_project_service.api_services
  create_duration = "${var.wait_seconds}s"
  triggers = {
    gce_api_id = each.value.id
  }
}
