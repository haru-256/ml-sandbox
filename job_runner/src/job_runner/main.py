from datetime import datetime
from typing import Literal, Optional

from google.cloud import aiplatform
from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict


class JobRunner(BaseSettings):
    """JobRunner class to run a job on Google Cloud AI Platform"""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_prog_name="job-runner",
        env_file=".env.local",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )

    # from environmental variables
    # FIXME: これも必須parameterになり、CLIから指定する必要があるように見える
    project: str = Field(description="[ENV] Google Cloud project ID to use for the job")
    location: str = Field(description="[ENV] Google Cloud region to use for the job")
    image_uri: str = Field(
        description="[ENV] Google Cloud Container Registry URI to use for the job"
    )
    gcs_path: str = Field(description="[ENV] Google Cloud Storage path to the model")
    service_account: str = Field(
        description="[ENV] Google Cloud service account to use for the job"
    )
    experiment_name: str = Field(description="[ENV] Experiment name to use for the job")

    # from args
    command: Optional[str] = Field(default=None, description="Command to run in the container")
    machine_type: Literal["g2-instance-4", "g2-instance-8"] = Field(
        description="Machine type to use for the job", default="g2-instance-4"
    )
    accelerator_type: Literal["NVIDIA_L4"] = Field(
        description="Accelerator type to use for the job", default="NVIDIA_L4"
    )
    accelerator_count: int = Field(
        description="Number of accelerators to use for the job", default=1
    )

    @field_validator("gcs_path", mode="after")
    @classmethod
    def validate_gcs_path(cls, v: str) -> str:
        """Validate the GCS path"""
        if not v.startswith("gs://"):
            raise ValueError("GCS path must start with 'gs://', but got: " + v)
        return v

    def cli_cmd(self) -> None:
        logger.info("Running the job-runner cli command")
        logger.info(self.model_dump())

        # parse command
        if self.command is not None:
            command = self.command.split(" ")
        else:
            command = None

        suffix = datetime.now().strftime("%Y%m%d%H%M%S")
        job = aiplatform.CustomContainerTrainingJob(
            display_name=f"{self.experiment_name}_{suffix}",
            project=self.project,
            location=self.location,
            container_uri=self.image_uri,
            staging_bucket=self.gcs_path,
            # FIXME: If the following PR is merged, this will be removed
            # https://github.com/googleapis/python-aiplatform/pull/5162
            command=command,  # type: ignore
        )
        job.run(
            service_account=self.service_account,
            machine_type=self.machine_type,
            accelerator_type=self.accelerator_type,
            accelerator_count=self.accelerator_count,
            replica_count=1,
            base_output_dir=self.gcs_path,
            disable_retries=True,
            environment_variables={
                "PROJECT": self.project,
                "EXPERIMENT_NAME": self.experiment_name,
                "GCS_PATH": self.gcs_path,
            },
        )


def main() -> None:
    """Main function to run the job-runner"""
    CliApp.run(JobRunner)


if __name__ == "__main__":
    main()
