import re
from typing import Any

import pytest
from pytest_console_scripts import ScriptRunner
from pytest_mock import MockerFixture

from vertex_job_runner.job import run_custom_training_job
from vertex_job_runner.settings import Settings

PRG = "vrun"


def strip_ansi(text: str) -> str:
    """
    Remove ANSI escape codes from a string.

    This is useful for testing CLI output where color codes might be present
    depending on the environment (e.g., CI vs local).
    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def make_settings(**overrides: Any) -> Settings:
    """Create Settings for job unit tests."""
    values: dict[str, Any] = {
        "project": "test-project",
        "location": "us-central1",
        "image_uri": "us-docker.pkg.dev/test/image:latest",
        "gcs_uri": "gs://test-bucket/vertex/",
        "service_account": "trainer@test-project.iam.gserviceaccount.com",
        "experiment_name": "test-experiment",
        "command": ["uv", "run", "python", "src/fit.py"],
        "machine_type": "g2-standard-4",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 1,
        "args": ["model=SASRec"],
    }
    values.update(overrides)
    return Settings(**values)


def test_help(script_runner: ScriptRunner) -> None:
    """Test that the CLI prints usage information for the --help flag."""
    result = script_runner.run([PRG, "--help"], check=True)
    assert re.search(rf"Usage: {PRG}", strip_ansi(result.stdout)) is not None


def test_dry_run_config_loading(script_runner: ScriptRunner) -> None:
    """Test that dry-run mode loads and displays values from pyproject.toml and environment variables."""
    result = script_runner.run([PRG, "--dry-run"], check=True)
    # Check if command is correctly split into a list in the output table
    assert "['uv', 'run', 'main.py']" in result.stdout
    assert "['hoge=2', 'fuga=1']" in result.stdout
    assert "haru256-vertex-ai-sandbox" in result.stdout


def test_cli_args_override(script_runner: ScriptRunner) -> None:
    """Test that CLI --args overrides the default args in the dry-run output."""
    result = script_runner.run([PRG, "--args", "param1=val1 param2=val2", "--dry-run"], check=True)
    assert "['param1=val1', 'param2=val2']" in result.stdout


def test_run_custom_training_job_passes_wandb_api_key_without_gcs_env(
    mocker: MockerFixture,
) -> None:
    """Test that WANDB_API_KEY is passed to the container and GCS_URI is not.

    Also verifies that gcs_uri is still supplied to the Vertex AI SDK constructor
    and run() kwargs for staging_bucket and base_output_dir.
    """
    settings = make_settings(wandb_api_key="secret-wandb-key")
    mock_job = mocker.MagicMock()
    mock_job.resource_name = "projects/test/locations/us-central1/trainingPipelines/123"

    mock_cls = mocker.patch("vertex_job_runner.job.aiplatform.CustomContainerTrainingJob")
    mock_cls.return_value = mock_job
    resource_name = run_custom_training_job(settings)

    assert resource_name == "projects/test/locations/us-central1/trainingPipelines/123"
    _, ctor_kwargs = mock_cls.call_args
    assert ctor_kwargs["staging_bucket"] == "gs://test-bucket/vertex/"
    assert ctor_kwargs["container_uri"] == "us-docker.pkg.dev/test/image:latest"
    _, run_kwargs = mock_job.run.call_args
    assert run_kwargs["base_output_dir"] == "gs://test-bucket/vertex/"
    assert run_kwargs["environment_variables"] == {
        "PROJECT": "test-project",
        "EXPERIMENT_NAME": "test-experiment",
        "WANDB_API_KEY": "secret-wandb-key",
    }
    assert "GCS_URI" not in run_kwargs["environment_variables"]


def test_run_custom_training_job_omits_wandb_api_key_when_unset(
    mocker: MockerFixture,
) -> None:
    """Test that WANDB_API_KEY is omitted from container env vars when not configured."""
    settings = make_settings()
    mock_job = mocker.MagicMock()
    mock_job.resource_name = "projects/test/locations/us-central1/trainingPipelines/456"

    mock_cls = mocker.patch("vertex_job_runner.job.aiplatform.CustomContainerTrainingJob")
    mock_cls.return_value = mock_job
    run_custom_training_job(settings)

    _, ctor_kwargs = mock_cls.call_args
    assert ctor_kwargs["staging_bucket"] == "gs://test-bucket/vertex/"
    _, run_kwargs = mock_job.run.call_args
    assert run_kwargs["base_output_dir"] == "gs://test-bucket/vertex/"
    assert run_kwargs["environment_variables"] == {
        "PROJECT": "test-project",
        "EXPERIMENT_NAME": "test-experiment",
    }


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that VRUN_ prefixed environment variables override pyproject.toml defaults."""
    monkeypatch.setenv("VRUN_PROJECT", "env-project")
    monkeypatch.setenv("VRUN_MACHINE_TYPE", "n1-standard-8")
    monkeypatch.setenv("VRUN_ACCELERATOR_COUNT", "2")

    settings = Settings()

    assert settings.project == "env-project"
    assert settings.machine_type == "n1-standard-8"
    assert settings.accelerator_count == 2
