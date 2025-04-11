import hydra
from loguru import logger
from ml_sandbox_libs.utils import setup_logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logger()
    logger.info(f"Starting the fit process with configuration: {cfg}")


if __name__ == "__main__":
    main()
