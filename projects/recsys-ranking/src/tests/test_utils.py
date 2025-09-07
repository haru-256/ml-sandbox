from enum import StrEnum

import pytest

from utils import enum_from_str


class ExampleStrEnum(StrEnum):
    """Test enum for testing enum_from_str function."""

    OPTION_A = "option_a"
    OPTION_B = "option_b"
    OPTION_C = "option_c"
    COMPLEX_OPTION = "complex-option-with-dashes"
    NUMERIC_LIKE = "123"


class _SingleValueStrEnum(StrEnum):
    """Single value enum for edge case testing."""

    ONLY_VALUE = "only_value"


class TestEnumFromStr:
    """Test suite for the enum_from_str utility function."""

    def test_enum_from_str_with_valid_member_name(self) -> None:
        """Test enum_from_str with valid enum member names."""
        result = enum_from_str(ExampleStrEnum, "OPTION_A")
        assert result == ExampleStrEnum.OPTION_A
        assert isinstance(result, ExampleStrEnum)

    def test_enum_from_str_with_valid_value(self) -> None:
        """Test enum_from_str with valid enum values."""
        result = enum_from_str(ExampleStrEnum, "option_a")
        assert result == ExampleStrEnum.OPTION_A
        assert isinstance(result, ExampleStrEnum)

    def test_enum_from_str_with_complex_value(self) -> None:
        """Test enum_from_str with complex string values."""
        result = enum_from_str(ExampleStrEnum, "complex-option-with-dashes")
        assert result == ExampleStrEnum.COMPLEX_OPTION
        assert isinstance(result, ExampleStrEnum)

    def test_enum_from_str_with_numeric_like_value(self) -> None:
        """Test enum_from_str with numeric-like string values."""
        result = enum_from_str(ExampleStrEnum, "123")
        assert result == ExampleStrEnum.NUMERIC_LIKE
        assert isinstance(result, ExampleStrEnum)

    def test_enum_from_str_with_all_valid_members(self) -> None:
        """Test enum_from_str with all valid enum members."""
        test_cases = [
            ("OPTION_A", ExampleStrEnum.OPTION_A),
            ("OPTION_B", ExampleStrEnum.OPTION_B),
            ("OPTION_C", ExampleStrEnum.OPTION_C),
            ("COMPLEX_OPTION", ExampleStrEnum.COMPLEX_OPTION),
            ("NUMERIC_LIKE", ExampleStrEnum.NUMERIC_LIKE),
        ]

        for member_name, expected_value in test_cases:
            result = enum_from_str(ExampleStrEnum, member_name)
            assert result == expected_value, f"Failed for member name: {member_name}"

    def test_enum_from_str_with_all_valid_values(self) -> None:
        """Test enum_from_str with all valid enum values."""
        test_cases = [
            ("option_a", ExampleStrEnum.OPTION_A),
            ("option_b", ExampleStrEnum.OPTION_B),
            ("option_c", ExampleStrEnum.OPTION_C),
            ("complex-option-with-dashes", ExampleStrEnum.COMPLEX_OPTION),
            ("123", ExampleStrEnum.NUMERIC_LIKE),
        ]

        for value, expected_enum in test_cases:
            result = enum_from_str(ExampleStrEnum, value)
            assert result == expected_enum, f"Failed for value: {value}"

    def test_enum_from_str_with_none_input(self) -> None:
        """Test enum_from_str returns None when input is None."""
        result = enum_from_str(ExampleStrEnum, None)
        assert result is None

    def test_enum_from_str_with_invalid_string(self) -> None:
        """Test enum_from_str raises ValueError for invalid strings."""
        with pytest.raises(ValueError) as exc_info:
            enum_from_str(ExampleStrEnum, "invalid_value")

        assert "'invalid_value' is not a valid ExampleStrEnum name or value" in str(exc_info.value)

    def test_enum_from_str_with_empty_string(self) -> None:
        """Test enum_from_str raises ValueError for empty string."""
        with pytest.raises(ValueError) as exc_info:
            enum_from_str(ExampleStrEnum, "")

        assert "'' is not a valid ExampleStrEnum name or value" in str(exc_info.value)

    def test_enum_from_str_with_whitespace_string(self) -> None:
        """Test enum_from_str raises ValueError for whitespace strings."""
        test_cases = [" ", "  ", "\t", "\n", " \t\n "]

        for whitespace in test_cases:
            with pytest.raises(ValueError) as exc_info:
                enum_from_str(ExampleStrEnum, whitespace)

            # Check that the error message contains the escaped representation for special chars
            expected_repr = repr(whitespace)[1:-1]  # Remove outer quotes
            assert f"'{expected_repr}' is not a valid ExampleStrEnum name or value" in str(
                exc_info.value
            )

    def test_enum_from_str_case_sensitivity(self) -> None:
        """Test enum_from_str is case sensitive."""
        # These should fail because of case sensitivity
        invalid_cases = ["option_A", "Option_A", "OPTION_a", "option_B", "Option_b"]

        for invalid_case in invalid_cases:
            with pytest.raises(ValueError) as exc_info:
                enum_from_str(ExampleStrEnum, invalid_case)

            assert f"'{invalid_case}' is not a valid ExampleStrEnum name or value" in str(
                exc_info.value
            )

    def test_enum_from_str_with_single_value_enum(self) -> None:
        """Test enum_from_str works with single-value enums."""
        # Test with member name
        result = enum_from_str(_SingleValueStrEnum, "ONLY_VALUE")
        assert result == _SingleValueStrEnum.ONLY_VALUE

        # Test with value
        result = enum_from_str(_SingleValueStrEnum, "only_value")
        assert result == _SingleValueStrEnum.ONLY_VALUE

    def test_enum_from_str_with_none_for_single_enum(self) -> None:
        """Test enum_from_str returns None for None input with single-value enum."""
        result = enum_from_str(_SingleValueStrEnum, None)
        assert result is None

    def test_enum_from_str_priority_member_over_value(self) -> None:
        """Test that member names take priority over values when they conflict."""

        class ConflictingStrEnum(StrEnum):
            OPTION_A = "OPTION_B"  # Member name OPTION_A has value "OPTION_B"
            OPTION_B = "option_b"

        # When we pass "OPTION_B", it should match the member name, not the value
        result = enum_from_str(ConflictingStrEnum, "OPTION_B")
        assert result == ConflictingStrEnum.OPTION_B
        assert result.value == "option_b"

        # When we pass "OPTION_A", it should match the member name
        result = enum_from_str(ConflictingStrEnum, "OPTION_A")
        assert result == ConflictingStrEnum.OPTION_A
        assert result.value == "OPTION_B"

    def test_enum_from_str_error_message_includes_enum_name(self) -> None:
        """Test that error messages include the correct enum class name."""
        with pytest.raises(ValueError) as exc_info:
            enum_from_str(ExampleStrEnum, "invalid")
        assert "ExampleStrEnum" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            enum_from_str(_SingleValueStrEnum, "invalid")
        assert "_SingleValueStrEnum" in str(exc_info.value)

    def test_enum_from_str_return_types(self) -> None:
        """Test that return types are correct for type hints."""
        # Test successful case
        result = enum_from_str(ExampleStrEnum, "OPTION_A")
        assert isinstance(result, ExampleStrEnum)

        # Test None case
        result = enum_from_str(ExampleStrEnum, None)
        assert result is None

    def test_enum_from_str_with_unicode_strings(self) -> None:
        """Test enum_from_str with unicode characters."""

        class UnicodeStrEnum(StrEnum):
            UNICODE_OPTION = "café"
            EMOJI_OPTION = "🚀"
            CHINESE_OPTION = "选项"

        # Test unicode values
        result = enum_from_str(UnicodeStrEnum, "café")
        assert result == UnicodeStrEnum.UNICODE_OPTION

        result = enum_from_str(UnicodeStrEnum, "🚀")
        assert result == UnicodeStrEnum.EMOJI_OPTION

        result = enum_from_str(UnicodeStrEnum, "选项")
        assert result == UnicodeStrEnum.CHINESE_OPTION

        # Test member names
        result = enum_from_str(UnicodeStrEnum, "UNICODE_OPTION")
        assert result == UnicodeStrEnum.UNICODE_OPTION

    def test_enum_from_str_preserves_original_exception_chain(self) -> None:
        """Test that the original ValueError is preserved in the exception chain."""
        with pytest.raises(ValueError) as exc_info:
            enum_from_str(ExampleStrEnum, "invalid_value")

        # The exception should have a cause (the original ValueError from enum construction)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_enum_from_str_comprehensive_error_scenarios(self) -> None:
        """Test various error scenarios to ensure robust error handling."""
        error_test_cases = [
            ("123invalid", "not a valid ExampleStrEnum"),
            ("option_a_extra", "not a valid ExampleStrEnum"),
            ("OPTION_A_EXTRA", "not a valid ExampleStrEnum"),
            ("option-a", "not a valid ExampleStrEnum"),  # Dash instead of underscore
            ("oPTION_A", "not a valid ExampleStrEnum"),  # Mixed case
        ]

        for invalid_input, expected_error_fragment in error_test_cases:
            with pytest.raises(ValueError) as exc_info:
                enum_from_str(ExampleStrEnum, invalid_input)

            assert expected_error_fragment in str(exc_info.value)
            assert invalid_input in str(exc_info.value)

    def test_enum_from_str_with_special_characters_in_enum_values(self) -> None:
        """Test enum_from_str with enum values containing special characters."""

        class SpecialCharStrEnum(StrEnum):
            SPACE_VALUE = "value with spaces"
            NEWLINE_VALUE = "value\nwith\nnewlines"
            TAB_VALUE = "value\twith\ttabs"
            QUOTE_VALUE = 'value"with"quotes'
            MIXED_VALUE = "value!@#$%^&*()_+-=[]{}|;':\",./<>?"

        # Test all special character values work correctly
        assert (
            enum_from_str(SpecialCharStrEnum, "value with spaces") == SpecialCharStrEnum.SPACE_VALUE
        )
        assert (
            enum_from_str(SpecialCharStrEnum, "value\nwith\nnewlines")
            == SpecialCharStrEnum.NEWLINE_VALUE
        )
        assert (
            enum_from_str(SpecialCharStrEnum, "value\twith\ttabs") == SpecialCharStrEnum.TAB_VALUE
        )
        assert (
            enum_from_str(SpecialCharStrEnum, 'value"with"quotes') == SpecialCharStrEnum.QUOTE_VALUE
        )
        assert (
            enum_from_str(SpecialCharStrEnum, "value!@#$%^&*()_+-=[]{}|;':\",./<>?")
            == SpecialCharStrEnum.MIXED_VALUE
        )

        # Test member names also work
        assert enum_from_str(SpecialCharStrEnum, "SPACE_VALUE") == SpecialCharStrEnum.SPACE_VALUE
        assert enum_from_str(SpecialCharStrEnum, "MIXED_VALUE") == SpecialCharStrEnum.MIXED_VALUE
