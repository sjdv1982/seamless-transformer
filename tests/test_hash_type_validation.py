import pytest

from seamless import Buffer
from seamless.checksum.hash_type_validation import HashTypeValidationError
from seamless.transformer import delayed
from seamless_transformer.transformation_class import TransformationError


def test_concrete_checksum_input_rejects_incompatible_hash_type():
    @delayed
    def text_length(value: str) -> int:
        return len(value)

    text_length.local = True
    checksum = Buffer(b"\xff\xfe\x00").get_checksum()

    with pytest.raises(HashTypeValidationError, match="Cannot deserialize"):
        text_length(checksum)


def test_transformation_future_input_rejects_incompatible_hash_type_after_resolution():
    @delayed
    def produce_bytes() -> bytes:
        return b"\xff\xfe\x00"

    @delayed
    def text_length(value: str) -> int:
        return len(value)

    produce_bytes.local = True
    produce_bytes.celltypes.result = "bytes"
    text_length.local = True
    text_length.celltypes.value = "text"
    text_length.celltypes.result = "int"

    with pytest.raises(TransformationError) as exc_info:
        text_length(produce_bytes()).run()

    assert "Cannot deserialize" in str(exc_info.value)
