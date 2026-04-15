import logging


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )
