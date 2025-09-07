from enum import StrEnum


def enum_from_str[T: StrEnum](cls: type[T], s: str | None) -> T | None:
    if s is None:
        return None
    if s in cls.__members__:
        return cls.__members__[s]
    try:
        return cls(s)
    except ValueError as e:
        raise ValueError(f"{s!r} is not a valid {cls.__name__} name or value") from e
