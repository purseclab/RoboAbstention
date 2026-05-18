from __future__ import annotations

from enum import IntEnum


class SizeOrder(IntEnum):
    xsmall = 0
    small = 1
    medium = 2
    large = 3
    xlarge = 4


def parse_size(value: str) -> SizeOrder:
    normalized = value.strip().lower()
    try:
        return SizeOrder[normalized]
    except KeyError as exc:
        raise ValueError(f"Invalid size value: {value!r}") from exc
