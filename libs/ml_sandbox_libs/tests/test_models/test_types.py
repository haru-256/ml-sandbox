from enum import StrEnum

import pytest

from ml_sandbox_libs.models.types import LinearOpOrderType, enum_from_str


class ExampleEnum(StrEnum):
    OPTION_A = "option_a"
    OPTION_B = "option_b"
    DASHED = "complex-option"


class ConflictingEnum(StrEnum):
    OPTION_A = "OPTION_B"
    OPTION_B = "option_b"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("OPTION_A", ExampleEnum.OPTION_A),
        ("option_b", ExampleEnum.OPTION_B),
        ("complex-option", ExampleEnum.DASHED),
    ],
)
def test_enum_from_str_accepts_member_names_and_values(
    raw_value: str, expected: ExampleEnum
) -> None:
    assert enum_from_str(ExampleEnum, raw_value) is expected


def test_enum_from_str_returns_none_for_none() -> None:
    assert enum_from_str(ExampleEnum, None) is None


def test_enum_from_str_prioritizes_member_name_over_value() -> None:
    assert enum_from_str(ConflictingEnum, "OPTION_B") is ConflictingEnum.OPTION_B


def test_enum_from_str_raises_informative_error() -> None:
    with pytest.raises(
        ValueError, match="'invalid' is not a valid ExampleEnum name or value"
    ) as exc_info:
        enum_from_str(ExampleEnum, "invalid")

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_linear_op_order_split_to_list() -> None:
    assert LinearOpOrderType.NORM_ACT_DROPOUT.split_to_list() == ("norm", "act", "dropout")
    assert LinearOpOrderType.ACT_NORM_DROPOUT.split_to_list() == ("act", "norm", "dropout")
