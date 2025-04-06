import json
import pathlib
import time
from typing import Literal, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Checkpoint
from lightning.pytorch.trainer.states import TrainerFn
from loguru import logger
from typing_extensions import override


class CustomModelCheckpoint(Checkpoint):
    def __init__(
        self,
        monitor: str,
        mode: Literal["min", "max"],
        every_n_epochs: int = 1,
        save_top_k: int = -1,
    ):
        """Custom ModelCheckpoint callback, GCS fuseに保存するとファイルシステムの違いでエラーが出るため、custom化
        issue: https://github.com/Lightning-AI/pytorch-lightning/issues/20270

        Args:
            monitor: _description_
            mode: _description_
            every_n_epochs: _description_. Defaults to 1.
            save_top_k: _description_. Defaults to -1.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        super().__init__()

        if not (save_top_k > 0 or save_top_k == -1):
            raise ValueError("save_top_k should be positive or -1")
        if mode not in ["min", "max"]:
            raise ValueError("mode should be 'min' or 'max'")

        self.monitor = monitor
        self.mode = mode
        self.every_n_epochs = every_n_epochs
        self.save_top_k = save_top_k

        self._checkpoints: list[tuple[float, pathlib.Path]] = []
        self._best_model_path: Optional[pathlib.Path] = None
        self._best_model_score: Optional[float] = None

    def _should_skip_saving_checkpoints(self, trainer: L.Trainer) -> bool:
        """Determine if checkpoint saving should be skipped.

        Args:
            trainer (L.Trainer): Lightning trainer instance.

        Returns:
            bool: True if saving should be skipped, False otherwise.
        """
        # from: lightning.pytorch.callbacks.model_checkpoint::ModelCheckpoint::_should_skip_saving_checkpoints
        return (
            bool(trainer.fast_dev_run)  # type: ignore
            or trainer.state.fn != TrainerFn.FITTING
            or trainer.sanity_checking
        )

    @property
    def best_model_path(self) -> Optional[pathlib.Path]:
        """Get the path to the best model checkpoint.

        Returns:
            Optional[pathlib.Path]: Path to the best model checkpoint.

        Raises:
            ValueError: If no best model path has been set.
        """
        if self._best_model_path is None:
            raise ValueError("No best model path found")
        return self._best_model_path

    @property
    def best_model_score(self) -> Optional[float]:
        """Get the score of the best model.

        Returns:
            Optional[float]: Score of the best model.

        Raises:
            ValueError: If no best model score has been set.
        """
        if self._best_model_score is None:
            raise ValueError("No best model score found")
        return self._best_model_score

    def _sorted_checkpoints(self) -> list[tuple[float, pathlib.Path]]:
        """Sort checkpoints based on their scores.

        Returns:
            list[tuple[float, pathlib.Path]]: List of (score, path) tuples sorted by score.
        """
        return sorted(self._checkpoints, key=lambda x: x[0], reverse=self.mode == "max")

    def _update_best_model(self) -> None:
        """Update the best model path and score based on current checkpoints."""
        if len(self._checkpoints) == 0:
            return

        sorted_checkpoints = self._sorted_checkpoints()
        self._best_model_score, self._best_model_path = sorted_checkpoints[0]

    def get_top_k_checkpoints(self) -> list[tuple[float, pathlib.Path]]:
        """Get the top K checkpoints based on their scores.

        Returns:
            list[tuple[float, pathlib.Path]]: List of top K (score, path) tuples.
        """
        if self.save_top_k == -1:
            return self._sorted_checkpoints()
        return self._sorted_checkpoints()[: self.save_top_k]

    def _clean_checkpoints(self) -> None:
        """Remove checkpoints that are not in the top K.

        Does nothing if save_top_k is -1 (keep all checkpoints).
        """
        if self.save_top_k == -1:
            return

        sorted_checkpoints = self._sorted_checkpoints()
        deleted_checkpoints = sorted_checkpoints[self.save_top_k :]

        for _, checkpoint in deleted_checkpoints:
            if checkpoint.exists():
                checkpoint.unlink()

    def _get_trainer_log_dir(self, trainer: L.Trainer) -> pathlib.Path:
        if trainer.logger is None:
            raise ValueError("Trainer logger is None")
        log_dir = trainer.logger.log_dir
        if log_dir is None:
            raise ValueError("Trainer logger has no log directory")
        return pathlib.Path(log_dir)

    @override
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save model checkpoints at the end of each training epoch.

        This method is called at the end of each training epoch to:
        1. Skip checkpoint saving if conditions are met (fast_dev_run, sanity_checking)
        2. Save model checkpoint if the current epoch matches the save frequency
        3. Update the best model tracking based on the monitored metric

        Args:
            trainer (L.Trainer): The Lightning trainer instance.
            pl_module (L.LightningModule): The Lightning module being trained.

        Side Effects:
            - Creates checkpoint directory if it doesn't exist
            - Saves model checkpoint to disk
            - Updates internal checkpoint tracking
            - Updates best model path and score
        """
        if self._should_skip_saving_checkpoints(trainer):
            return
        if not (self.every_n_epochs >= 1 and trainer.current_epoch % self.every_n_epochs == 0):
            return

        # Retrieval the monitored metric
        metrics = trainer.callback_metrics.get(self.monitor)
        if metrics is None:
            raise ValueError(f"Metric '{self.monitor}' not found in callback metrics")
        num_epochs = trainer.current_epoch
        num_steps = trainer.global_step

        # Make sure the save directory exists
        log_dir = self._get_trainer_log_dir(trainer)
        save_dir = pathlib.Path(log_dir) / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the checkpoint
        checkpoint_path = (
            save_dir
            / f"epoch={num_epochs:02d}-step={num_steps:010d}-{self.monitor}={metrics:.6g}.ckpt"
        )
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        _start = time.perf_counter()
        ## from trainer.save_checkpoint(checkpoint_path )
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved in {time.perf_counter() - _start:.2f} seconds")

        # update the best model
        self._checkpoints.append((float(metrics), checkpoint_path))
        self._update_best_model()
        self._clean_checkpoints()
        # breakpoint()

    @override
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Finalize the model checkpoint tracking at the end of training.

        This method is called once at the end of training to:
        1. Skip if in fast_dev_run or sanity checking mode
        2. Update the best model tracking
        3. Save the best model information to a JSON file

        Args:
            trainer (L.Trainer): The Lightning trainer instance.
            pl_module (L.LightningModule): The Lightning module that was trained.

        Side Effects:
            - Creates a 'best_model.json' file containing:
                - Monitored metric name
                - Monitor mode (min/max)
                - Top K model information (rank, metric value, file paths)
        """
        if self._should_skip_saving_checkpoints(trainer):
            logger.info("Skip saving checkpoints because of fast_dev_run or sanity_checking")
            return

        self._update_best_model()
        logger.info(f"Best model score: {self.best_model_score:.6g} at {self.best_model_path}")
        log_dir = self._get_trainer_log_dir(trainer)
        with open(log_dir / "best_model.json", "w") as f:
            json.dump(
                {
                    "monitor_metric": self.monitor,
                    "monitor_mode": self.mode,
                    "top_k_models": [
                        {
                            "rank": rnk + 1,
                            "metric": metric,
                            "file_name": checkpoint.name,
                            "path": str(checkpoint),
                        }
                        for rnk, (metric, checkpoint) in enumerate(self.get_top_k_checkpoints())
                    ],
                    "best_model_score": self.best_model_score,
                    "best_model_path": str(self.best_model_path),
                },
                f,
                indent=4,
            )
