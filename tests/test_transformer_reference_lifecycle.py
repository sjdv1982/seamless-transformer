import uuid

import pytest

from seamless import Buffer
from seamless.transformer import delayed
from seamless.caching import buffer_writer
from seamless.caching.buffer_cache import get_buffer_cache


def identity(value):
    return value


@pytest.fixture
def writes(monkeypatch):
    written = []
    monkeypatch.setattr(
        buffer_writer, "register", lambda buf: written.append(buf.get_checksum())
    )
    return written


def test_checksum_pin_is_refheld_and_replaced():
    first = Buffer(b"builder first").get_checksum()
    second = Buffer(b"builder second").get_checksum()
    transformer = delayed(identity)
    transformer.args.value = first
    cache = get_buffer_cache()
    assert cache.reference_snapshot()[first][0] == 1
    transformer.args.value = second
    assert cache.reference_snapshot().get(first, (0, 0, False))[0] == 0
    assert cache.reference_snapshot()[second][0] == 1
    transformer._release_refholds()
    assert cache.reference_snapshot().get(second, (0, 0, False))[0] == 0


def test_literal_pin_is_serialized_and_checksum_refheld():
    transformer = delayed(identity)
    transformer.args.value = "ordinary literal"
    checksum = Buffer("ordinary literal", "mixed").get_checksum()
    assert transformer._refheld_checksums() == [(checksum, "pin:value")]
    assert get_buffer_cache().reference_snapshot()[checksum][0] == 1
    transformer._release_refholds()
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0


@pytest.mark.parametrize("scratch", [False, True])
def test_literal_pin_is_published_whatever_the_transformer_scratch(writes, scratch):
    # pins.md, *Scratch at the pin*: a literal pin publishes regardless of
    # the transformer's result scratch.
    transformer = delayed(identity)
    transformer.scratch = scratch
    literal = f"published literal {uuid.uuid4().hex}"
    transformer.pins.value = literal
    checksum = Buffer(literal, "mixed").get_checksum()
    try:
        assert (checksum, "pin:value") in transformer._refheld_checksums()
        assert checksum in writes, "a literal pin was not published"
    finally:
        transformer._release_refholds()


def test_args_deletion_releases_checksum_pin():
    checksum = Buffer(b"builder deletion").get_checksum()
    transformer = delayed("result = value")
    transformer.args.value = checksum
    del transformer.args.value
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0
    transformer._release_refholds()


def test_public_buffer_access_acquires_an_internally_published_result():
    transformer = delayed(identity)(1)
    result_buffer = Buffer(123, "int")
    result = result_buffer.get_checksum()
    transformer._publish_result(result)
    assert get_buffer_cache().reference_snapshot().get(result, (0, 0, False))[0] == 0
    assert transformer.buffer.get_value("int") == 123
    assert get_buffer_cache().reference_snapshot()[result][0] == 1
    transformer._release_refholds()
    assert get_buffer_cache().reference_snapshot().get(result, (0, 0, False))[0] == 0


def test_cancellation_releases_unstarted_transformation_roles():
    input_buffer = Buffer(12, "int")
    input_checksum = input_buffer.get_checksum()
    transformation = delayed(identity)(input_checksum)
    assert transformation.cancel() is True
    assert not transformation._refheld_checksums()
    assert get_buffer_cache().reference_snapshot().get(
        input_checksum, (0, 0, False)
    )[0] == 0


def test_existing_builder_copy_has_independent_registry_and_roles():
    checksum = Buffer(b"builder copy").get_checksum()
    original = delayed("result = value")
    original.args.value = checksum
    copied = delayed(original)
    assert copied is not original
    assert copied._args is not original._args
    assert get_buffer_cache().reference_snapshot()[checksum][0] == 2
    del copied.args.value
    assert get_buffer_cache().reference_snapshot()[checksum][0] == 1
    original._release_refholds()
    copied._release_refholds()
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0
