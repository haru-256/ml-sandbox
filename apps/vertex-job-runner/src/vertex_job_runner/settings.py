import shlex
from typing import Any, List, Optional, Tuple, Type

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PyprojectTomlConfigSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings for the Vertex AI Job Runner.

    Configuration is loaded from (in order of priority):
    1. CLI arguments
    2. Environment variables (prefix: VRUN_)
    3. pyproject.toml ([tool.vrun] table)
    """

    app_name: str = "vertex-job-runner"
    version: str = "0.1.0"

    # Items not overridden by arguments
    project: str = Field(default=..., description="The Google Cloud Project ID")
    location: str = Field(default=..., description="The Google Cloud region for the job")
    image_uri: str = Field(default=..., description="The URI of the Docker image to be used")
    gcs_uri: str = Field(default=..., description="The GCS URI for artifacts and logs")
    service_account: str = Field(
        default=..., description="The service account email to run the job"
    )
    experiment_name: str = Field(default=..., description="The name of the Vertex AI experiment")
    command: List[str] = Field(default=..., description="The command to run inside the container")

    # Items overridable by arguments
    machine_type: str = Field(
        default=..., description="The type of machine to use for the Vertex AI job"
    )
    accelerator_type: str = Field(
        default=..., description="The type of accelerator to use for the Vertex AI job"
    )
    accelerator_count: int = Field(default=..., description="The number of accelerators to use")
    args: Optional[List[str]] = Field(
        default=None, description="The list of arguments for the command"
    )

    model_config = SettingsConfigDict(
        env_prefix="VRUN_", pyproject_toml_table_header=("tool", "vrun")
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> Tuple[Any, ...]:
        """
        Define the priority of configuration sources.

        Priority:
        1. Init settings (CLI args)
        2. Environment variables
        3. pyproject.toml
        """
        return (
            init_settings,
            env_settings,
            PyprojectTomlConfigSettingsSource(settings_cls),
        )

    @field_validator("command", mode="before")
    @classmethod
    def validate_command(cls, v: str | List[str]) -> List[str]:
        """Validate and split the command if it's a string"""

        if isinstance(v, str):
            return shlex.split(v)
        return v

    @field_validator("gcs_uri", mode="after")
    @classmethod
    def validate_gcs_uri(cls, v: str) -> str:
        """Validate the GCS URI"""
        if not v.startswith("gs://"):
            raise ValueError("GCS URI must start with 'gs://', but got: " + v)
        return v
