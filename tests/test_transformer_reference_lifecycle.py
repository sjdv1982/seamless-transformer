from seamless import Buffer
from seamless.transformer import delayed
from seamless.caching.buffer_cache import get_buffer_cache


def identity(value):
    return value


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


def test_literal_pin_is_not_checksum_refheld():
    transformer = delayed(identity)
    transformer.args.value = "ordinary literal"
    assert not get_buffer_cache().refholder_counts
    transformer._release_refholds()

