from enum import IntEnum


class SpecialIndex(IntEnum):
    PAD = 0
    UNK = 1


# 評価時のネガティブサンプル数
EVAL_NEGATIVE_SAMPLE_SIZE = 100
