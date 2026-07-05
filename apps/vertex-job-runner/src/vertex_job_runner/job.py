from datetime import datetime

from google.cloud import aiplatform

from vertex_job_runner.settings import Settings


def run_custom_training_job(settings: Settings) -> str:
    """Run a custom training job on Vertex AI.

    The training container always receives `PROJECT` and `EXPERIMENT_NAME` as
    environment variables. If `settings.wandb_api_key` is set, it is injected as
    `WANDB_API_KEY`. `GCS_URI` is intentionally not passed to the container;
    `gcs_uri` is only used for Vertex AI SDK staging/output.

    Args:
        settings: The configured job settings.

    Returns:
        The Vertex AI training pipeline resource name.
    """
    suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    job = aiplatform.CustomContainerTrainingJob(
        display_name=f"{settings.experiment_name}_{suffix}",
        project=settings.project,
        location=settings.location,
        container_uri=settings.image_uri,
        staging_bucket=settings.gcs_uri,
        command=settings.command,
    )
    environment_variables = {
        "PROJECT": settings.project,
        "EXPERIMENT_NAME": settings.experiment_name,
    }
    if settings.wandb_api_key is not None:
        environment_variables["WANDB_API_KEY"] = settings.wandb_api_key.get_secret_value()

    job.run(
        service_account=settings.service_account,
        machine_type=settings.machine_type,
        accelerator_type=settings.accelerator_type,
        accelerator_count=settings.accelerator_count,
        replica_count=1,
        base_output_dir=settings.gcs_uri,
        disable_retries=True,
        args=settings.args,  # type: ignore
        environment_variables=environment_variables,
    )

    return job.resource_name
