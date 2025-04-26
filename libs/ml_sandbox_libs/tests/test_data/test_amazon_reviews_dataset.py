import polars as pl

from ml_sandbox_libs.data.amazon_reviews_dataset import unk_filter_by_count


def test_unk_filter_by_count() -> None:
    # Test with a simple example
    df = pl.from_dict({"id": [1] * 95 + [2] * 5})
    filtered_df = unk_filter_by_count(df, id_column_name="id", threshold=0.96)
    assert len(filtered_df) == 1
    assert filtered_df["id"].item() == 1
