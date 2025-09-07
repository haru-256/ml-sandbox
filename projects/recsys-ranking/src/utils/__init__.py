from enum import StrEnum


def enum_from_str[T: StrEnum](cls: type[T], s: str | None) -> T | None:
    """Converts a string to an enum member, supporting both names and values.

    This function attempts to convert a given string into a member of the specified
    StrEnum subclass. It first checks if the string matches an enum member's name,
    then tries to match it as a value. This provides flexibility by allowing
    conversion from either the symbolic name or the underlying string value of
    an enum member.

    Args:
        cls: The StrEnum subclass to convert the string to.
        s: The string to convert. Can be None.

    Returns:
        The corresponding enum member if a match is found, or None if the input
        string is None.

    Raises:
        ValueError: If the string does not correspond to any member's name or
            value in the given enum class.

    Example:
        class MyEnum(StrEnum):
            A = "value_a"
            B = "value_b"

        enum_from_str(MyEnum, "A")  # Returns MyEnum.A
        enum_from_str(MyEnum, "value_a")  # Returns MyEnum.A
        enum_from_str(MyEnum, "C")  # Raises ValueError
    """
    if s is None:
        return None
    if s in cls.__members__:
        return cls.__members__[s]
    try:
        return cls(s)
    except ValueError as e:
        raise ValueError(f"{s!r} is not a valid {cls.__name__} name or value") from e
