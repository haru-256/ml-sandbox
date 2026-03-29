import pathlib

from pytest_mock import MockerFixture

from ml_sandbox_libs.data.factory import create_seq_rec_datamodule


def test_create_seq_rec_datamodule_initializes_with_dataset_subdirectory(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    """Create the shared seq-rec datamodule under the base path's dataset subdirectory."""
    mock_datamodule = mocker.Mock()
    datamodule_cls = mocker.patch(
        "ml_sandbox_libs.data.factory.AmazonReviewsSeqRecDataModule",
        return_value=mock_datamodule,
    )

    result = create_seq_rec_datamodule(
        save_dir=tmp_path,
        batch_size=128,
        max_seq_len=50,
        neg_sample_size=4,
        num_workers=2,
        eval_negative_sample_size=99,
    )

    assert result is mock_datamodule
    datamodule_cls.assert_called_once_with(
        save_dir=tmp_path / "dataset",
        batch_size=128,
        max_seq_len=50,
        neg_sample_size=4,
        num_workers=2,
        eval_negative_sample_size=99,
    )
