output "tfstate_gcs_bucket_id" {
  value       = google_storage_bucket.tfstate_bucket.id
  description = "The ID of the bucket used to store terraform state"
}
