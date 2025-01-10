from loguru import logger
import polars as pl

from utils.utils import weighted_average_embedding_df

N = 20

if __name__ == "__main__":
    logger.info("init inputs")
    tgt_df = pl.DataFrame(
        {
            "id": [0],
            "src_list": [[{"id": 1, "weight": 0.1} for _ in range(30)]],
        }
    )
    tgt_df = pl.concat([tgt_df] * N)
    src_embedding_df = pl.DataFrame({"id": [1], "embedding": [[1.0] * 128]})
    src_embedding_df = pl.concat([src_embedding_df] * N)

    logger.info("start weighted_average_embedding_df")
    weighted_average_embedding_df(tgt_df, src_embedding_df)
