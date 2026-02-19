output "service_account_email" {
  value       = google_service_account.training_sa.email
  description = "The email of the service account used for Vertex AI Training"
}
