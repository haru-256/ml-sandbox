output "tfstate_bucket_id" {
  value       = module.tfstate_bucket.tfstate_gcs_bucket_id
  description = "The ID of the bucket used to store terraform state"
}

output "vertex_ai_custom_training_job_sa" {
  value       = module.vertex_ai_training.service_account_email
  description = "The email of the service account used for Vertex AI Training"
}
