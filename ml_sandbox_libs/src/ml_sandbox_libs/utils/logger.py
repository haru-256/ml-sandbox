import os
import pathlib
import sys
from typing import Literal, Optional

import google.cloud.logging
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from loguru import logger


def check_is_in_vertexai_training() -> bool:
    """
    Check if the current environment is Vertex AI Training.

    Returns:
        bool: True if in Vertex AI Training, False otherwise.
    """
    return os.environ.get("CLOUD_ML_JOB") is not None


def setup_logger(
    level: Literal["INFO", "DEBUG"] = "INFO",
    log_path: Optional[pathlib.Path] = None,
) -> None:
    """
    Setup the logger for the project.

    Args:
        level: logger level. Defaults to "INFO".
        log_path: saved path in logging. Defaults to None.
    """
    is_in_vertexai_training = check_is_in_vertexai_training()

    if log_path is None and not is_in_vertexai_training:
        return

    # Remove the default sink
    logger.remove()

    # Add a new sink for stdout
    logger.add(sys.stdout, level=level)
    # Add a sink fpr a log file
    if log_path is not None:
        logger.add(log_path, level=level)

    if is_in_vertexai_training:
        project = os.environ.get("PROJECT")
        if project is None:
            raise ValueError("PROJECT environment variable is not set.")
        experiment_name = os.environ.get("EXPERIMENT_NAME")
        if experiment_name is None:
            raise ValueError("EXPERIMENT_NAME environment variable is not set.")

        client = google.cloud.logging.Client(project=project)
        handler = CloudLoggingHandler(client, name=experiment_name)
        handler.setLevel(level)
        logger.add(handler)
