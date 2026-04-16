from __future__ import annotations

import gzip

import zstandard

COMPRESSION_SUFFIXES = (".zst", ".gz")
COMPRESSION_PREFERENCE = (".zst", ".gz")
CONTENT_ENCODING_TO_SUFFIX = {
    "zstd": ".zst",
    "gzip": ".gz",
}
SUFFIX_TO_CONTENT_ENCODING = {
    suffix: encoding for encoding, suffix in CONTENT_ENCODING_TO_SUFFIX.items()
}


def strip_compression_suffix(name: str) -> tuple[str, str | None]:
    for suffix in COMPRESSION_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return name, None


def decompress_bytes(data: bytes, suffix: str) -> bytes:
    if suffix == ".zst":
        return zstandard.ZstdDecompressor().decompress(data)
    if suffix == ".gz":
        return gzip.decompress(data)
    raise ValueError(suffix)


def compress_bytes(data: bytes, suffix: str) -> bytes:
    if suffix == ".zst":
        return zstandard.ZstdCompressor().compress(data)
    if suffix == ".gz":
        return gzip.compress(data)
    raise ValueError(suffix)


def content_encoding_to_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    return CONTENT_ENCODING_TO_SUFFIX.get(value.strip().lower())


def suffix_to_content_encoding(suffix: str | None) -> str | None:
    if suffix is None:
        return None
    return SUFFIX_TO_CONTENT_ENCODING.get(suffix)
