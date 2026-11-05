"""Checksum algorithms used by checkpoint delta formats."""

import zlib
from functools import lru_cache
from typing import Any


class _Adler32:
    def __init__(self):
        self._value = 1

    def update(self, data: Any) -> None:
        self._value = zlib.adler32(data, self._value)

    def hexdigest(self) -> str:
        return f"{self._value:08x}"


def create_checksum(algorithm: str):
    if algorithm == "xxh3-128":
        import xxhash

        return xxhash.xxh3_128()
    if algorithm == "blake3":
        import blake3

        return blake3.blake3()
    if algorithm == "adler32":
        return _Adler32()
    raise ValueError(f"unsupported checksum algorithm {algorithm!r}")


def calculate_checksum(algorithm: str, data: Any) -> str:
    checksum = create_checksum(algorithm)
    checksum.update(data)
    return checksum.hexdigest()


@lru_cache(maxsize=None)
def _checksum_hex_length(algorithm: str) -> int:
    return len(create_checksum(algorithm).hexdigest())


def validate_checksum(algorithm: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid {algorithm} checksum: {value!r}")
    expected_length = _checksum_hex_length(algorithm)
    if len(value) != expected_length:
        raise ValueError(f"invalid {algorithm} checksum: {value!r}")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"invalid {algorithm} checksum: {value!r}")
    return value.lower()
