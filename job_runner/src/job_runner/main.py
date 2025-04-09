import pathlib
from typing import Literal

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict


class JobRunner(BaseSettings):
    """Settings for the job-runner"""

    model_config = SettingsConfigDict(
        cli_parse_args=True, cli_prog_name="job-runner", env_file=".env", env_file_encoding="utf-8"
    )

    # from environmental variables
    # FIXME: これも必須parameterになり、CLIから指定する必要があるように見える
    gcs_path: pathlib.Path = Field(description="[ENV] Google Cloud Storage path to the model")

    # from args
    machine_type: Literal["g2-instance=4", "g2-instance=12"] = Field(
        description="Machine type to use for the job"
    )

    def cli_cmd(self) -> None:
        logger.info("Running the job-runner cli command")
        logger.info(self.model_dump())


def main() -> None:
    """Main function to run the job-runner"""
    CliApp.run(JobRunner)


if __name__ == "__main__":
    main()
