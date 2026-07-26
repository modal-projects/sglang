"""Checksum algorithms used by checkpoint delta formats."""

import zlib
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
