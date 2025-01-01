import pathlib
from copy import deepcopy
from typing import Optional
import json

import lightning as L
import torch
import pytest
from deepdiff import DeepDiff
from pytest_mock import MockerFixture
import time


from utils.callbacks import CustomModelCheckpoint
from contextlib import contextmanager
from loguru import logger


def _custom_model_checkpoint_factory(
    mode: str,
    save_top_k: Optional[int] = -1,
    checkpoints: Optional[list[tuple[float, pathlib.Path]]] = None,
) -> CustomModelCheckpoint:
    callback = CustomModelCheckpoint(monitor="metric", mode=mode, save_top_k=save_top_k)  # type: ignore
    if checkpoints is not None:
        callback._checkpoints = deepcopy(checkpoints)
    return callback


@contextmanager
def dummy_profiler():
    logger.info("in")
    yield
    logger.info("end")


class TestCustomModelCheckpoint:
    def test_invalid_save_top_k(self) -> None:
        with pytest.raises(ValueError, match="save_top_k should be positive or -1"):
            CustomModelCheckpoint(monitor="val_loss", mode="min", save_top_k=0)

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode should be 'min' or 'max'"):
            CustomModelCheckpoint(monitor="val_loss", mode="invalid")  # type: ignore

    def test_best_model_properties_before_setting(self) -> None:
        callback = _custom_model_checkpoint_factory(mode="min")
        with pytest.raises(ValueError, match="No best model path found"):
            _ = callback.best_model_path
        with pytest.raises(ValueError, match="No best model score found"):
            _ = callback.best_model_score

    def test_get_top_k_checkpoints(self) -> None:
        checkpoints = [
            (0.5, pathlib.Path("model1.ckpt")),
            (0.3, pathlib.Path("model2.ckpt")),
            (0.7, pathlib.Path("model3.ckpt")),
        ]
        callback = _custom_model_checkpoint_factory(mode="min", checkpoints=checkpoints)
        top_k = callback.get_top_k_checkpoints()
        assert len(top_k) == 3 and DeepDiff(top_k, checkpoints)

        callback = _custom_model_checkpoint_factory(
            mode="min", checkpoints=checkpoints, save_top_k=2
        )
        top_k = callback.get_top_k_checkpoints()
        assert len(top_k) == 2 and DeepDiff(top_k, [checkpoints[1], checkpoints[0]])

    def test_update_best_model(self) -> None:
        checkpoints = [
            (0.5, pathlib.Path("model1.ckpt")),
            (0.3, pathlib.Path("model2.ckpt")),
            (0.7, pathlib.Path("model3.ckpt")),
        ]

        callback = _custom_model_checkpoint_factory(mode="min", checkpoints=checkpoints)
        callback._update_best_model()
        assert callback._best_model_score == 0.3
        assert callback._best_model_path == pathlib.Path("model2.ckpt")

        # Test with max mode
        callback = _custom_model_checkpoint_factory(mode="max", checkpoints=checkpoints)
        callback._update_best_model()
        assert callback._best_model_score == 0.7
        assert callback._best_model_path == pathlib.Path("model3.ckpt")

        # Test with empty checkpoints
        callback = _custom_model_checkpoint_factory(mode="min", checkpoints=checkpoints)
        assert callback._best_model_score is None
        assert callback._best_model_path is None

    def test_get_trainer_log_dir(self, mocker: MockerFixture) -> None:
        callback = CustomModelCheckpoint(monitor="val_loss", mode="min")

        # Test with valid log dir
        mock_trainer = mocker.MagicMock(L.Trainer)
        mock_trainer.logger.log_dir = "test/path"
        log_dir = callback._get_trainer_log_dir(mock_trainer)
        assert log_dir == pathlib.Path("test/path")

        # Test with None logger
        mock_trainer.logger = None
        with pytest.raises(ValueError, match="Trainer logger is None"):
            callback._get_trainer_log_dir(mock_trainer)

        # Test with None log_dir
        mock_trainer.logger = mocker.MagicMock()
        mock_trainer.logger.log_dir = None
        with pytest.raises(ValueError, match="Trainer logger has no log directory"):
            callback._get_trainer_log_dir(mock_trainer)

    def test_on_train_epoch_end(self, mocker: MockerFixture, tmp_path: pathlib.Path) -> None:
        checkpoints = [
            (0.5, pathlib.Path("model1.ckpt")),
            (0.3, pathlib.Path("model2.ckpt")),
        ]

        # testするcallbackを作成
        callback = _custom_model_checkpoint_factory(
            mode="min", checkpoints=checkpoints, save_top_k=2
        )
        mocker.patch.object(
            callback, "_should_skip_saving_checkpoints", return_value=False, autospec=True
        )
        # DIするtrainerとlightning moduleはMagicMockでモック
        trainer = mocker.MagicMock(L.Trainer)
        trainer.logger.log_dir = str(tmp_path)
        trainer.current_epoch = 10
        trainer.global_step = 0
        trainer.callback_metrics = {"metric": torch.tensor(0.1)}
        data = {"state_dict": torch.tensor(0.1)}
        _checkpoint_connector = mocker.MagicMock()
        _checkpoint_connector.dump_checkpoint.return_value = data
        trainer._checkpoint_connector = _checkpoint_connector
        # NOTE: 以下だとAttributeError: Mock object has no attribute 'profiler'となる。おそらく _ prefixのメソッドのmethodをmockできないため
        # trainer._checkpoint_connector.dump_checkpoint.return_value = data
        pl_module = mocker.MagicMock(L.LightningModule)

        callback.on_train_epoch_end(trainer, pl_module)
        checkpoint_path = tmp_path / "checkpoints" / "epoch=10-step=0000000000-metric=0.1.ckpt"
        assert checkpoint_path.exists()
        assert torch.load(checkpoint_path, weights_only=True) == data
        torch.testing.assert_close(callback.best_model_score, 0.1)
        assert callback.best_model_path == checkpoint_path
        actual_top_k_path = [
            path for _, path in callback.get_top_k_checkpoints()
        ]  # 数値的に不安定なのでfileのみ比較
        assert DeepDiff(actual_top_k_path, [checkpoint_path, checkpoints[1]])

    def test_on_fit_end(self, mocker: MockerFixture) -> None:
        checkpoints = [
            (0.5, pathlib.Path("model1.ckpt")),
            (0.3, pathlib.Path("model2.ckpt")),
            (0.7, pathlib.Path("model3.ckpt")),
        ]
        callback = _custom_model_checkpoint_factory(
            mode="min", checkpoints=checkpoints, save_top_k=2
        )

        mocker.patch.object(
            callback, "_should_skip_saving_checkpoints", side_effect=[True, False], autospec=True
        )
        # 以下でもOK
        # callback._should_skip_saving_checkpoints = mocker.MagicMock(
        #     side_effect=[True, False], autospec=True
        # )
        open_mock = mocker.patch("builtins.open", new_callable=mocker.mock_open)
        # DIするobjectはMagicMockでモック
        trainer = mocker.MagicMock(L.Trainer)
        trainer.logger.log_dir = "tests/path"
        pl_module = mocker.MagicMock(L.LightningModule)

        # Test skip
        callback.on_fit_end(trainer, pl_module)
        assert open_mock.call_count == 0

        # Test save
        callback.on_fit_end(trainer, pl_module)
        assert open_mock.call_count == 1
        # TODO: 書き込むデータの検証
        # handle = open_mock()
        # data = {
        #     "monitor": "metric",
        #     "mode": "min",
        #     "top_k_models": [
        #         {
        #             "rank": rnk + 1,
        #             "metric": metric,
        #             "file_name": str(checkpoint),
        #             "path": str(checkpoint),
        #         }
        #         for rnk, (metric, checkpoint) in enumerate(
        #             [(0.3, "model2.ckpt"), (0.5, "model1.ckpt")]
        #         )
        #     ],
        # }
        # assert handle.write.assert_called_once_with(json.dumps(data))
