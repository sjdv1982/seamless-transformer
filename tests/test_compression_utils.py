from seamless_transformer.compression_utils import compress_bytes, decompress_bytes


def test_transformer_compression_utils_round_trip():
    payloads = [b"", b"abc", b"xyz" * 10000]
    for suffix in (".zst", ".gz"):
        for payload in payloads:
            assert decompress_bytes(compress_bytes(payload, suffix), suffix) == payload
