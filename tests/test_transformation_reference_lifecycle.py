from seamless import Buffer
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.transformer import delayed


def add_one(value):
    return value + 1


def test_transformation_keeps_input_definition_and_public_result_once():
    input_buffer = Buffer(10, "int")
    input_checksum = input_buffer.get_checksum()
    transformation = delayed(add_one)(input_checksum)
    result = transformation.compute()
    assert result is not None

    cache = get_buffer_cache()
    snapshot = cache.reference_snapshot()
    assert snapshot[input_checksum][0] >= 1
    assert snapshot[transformation.transformation_checksum][0] == 1
    assert snapshot[result][0] == 1
    assert transformation.compute() == result
    assert cache.reference_snapshot()[result][0] == 1

    transformation._release_refholds()
    assert cache.reference_snapshot().get(input_checksum, (0, 0, False))[0] == 0
    assert cache.reference_snapshot().get(
        transformation.transformation_checksum, (0, 0, False)
    )[0] == 0
    assert cache.reference_snapshot().get(result, (0, 0, False))[0] == 0


def test_internal_dependency_evaluation_does_not_hold_producer_result():
    producer = delayed(add_one)(10)
    consumer = delayed(add_one)(producer)
    result = consumer.compute()
    producer_result = producer._result_checksum_internal()
    assert result is not None
    assert producer_result is not None
    cache = get_buffer_cache()
    # The consumer adopts the input independently; producer result ownership is
    # not created by the scheduler's internal dependency path.
    assert cache.reference_snapshot()[producer_result][0] == 1
    producer._release_refholds()
    assert cache.reference_snapshot()[producer_result][0] == 1
    consumer._release_refholds()
    assert cache.reference_snapshot().get(producer_result, (0, 0, False))[0] == 0


def test_failed_transformation_publishes_no_result_reference():
    def fail(value):
        raise ValueError("expected failure")

    transformation = delayed(fail)(10)
    assert transformation.compute() is None
    assert transformation._result_checksum_internal() is None
    assert not any(
        role == "result" for _checksum, role in transformation._refheld_checksums()
    )
    transformation._release_refholds()

