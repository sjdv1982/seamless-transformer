import seamless.config
import pytest

seamless.config.init()

from seamless.transformer import direct, delayed
from seamless_dask.dummy_scheduler import create_dummy_client
from seamless_dask.transformer_client import set_seamless_dask_client

OFFSET = 25  # increment to force cache misses


def test_dependencies():
    import time

    offset = 1000 * OFFSET

    @delayed
    def slow_add_delayed(a, b, c=None, d=None, e=None) -> float:
        import time
        from seamless.transformer import global_lock

        with global_lock:
            time.sleep(1)
        return sum([x for x in (a, b, c, d, e) if x is not None])

    slow_add = direct(slow_add_delayed)

    @direct
    def fast_add(a, b, c=None, d=None, e=None):
        return sum([x for x in (a, b, c, d, e) if x is not None])

    assert fast_add(10, 20) == 30

    fast_add_delayed = delayed(fast_add)
    print()

    start = time.perf_counter()
    result = fast_add(slow_add_delayed(2 + offset, 3), slow_add_delayed(4 + offset, 5))
    print(result)
    assert result == 14 + 2 * offset, result
    duration = time.perf_counter() - start
    print(f"{duration:.3f}")

    start = time.perf_counter()
    result = fast_add(
        slow_add_delayed(2 + offset, 13), slow_add_delayed(4 + offset, 15)
    )
    assert result == 34 + 2 * offset, result
    duration = time.perf_counter() - start
    print(f"{duration:.3f}")

    start = time.perf_counter()
    result = slow_add(slow_add_delayed(5 + offset, 6), fast_add_delayed(1 + offset, 2))
    assert result == 14 + 2 * offset, result
    duration = time.perf_counter() - start
    print(f"{duration:.3f}")

    start = time.perf_counter()
    v1 = slow_add_delayed(9, -1 + offset, -2, -3)
    v2 = slow_add_delayed(10, -1 + offset, -2, -3)
    v3 = slow_add_delayed(11, -1 + offset, -2, -3)
    v4 = slow_add_delayed(12, -1 + offset, -2, -3)
    v5 = slow_add_delayed(13, -1 + offset, -2, -3)
    result = fast_add(v1, v2, v3, v4, v5)
    assert result == 3 + 4 + 5 + 6 + 7 + 5 * offset
    duration = time.perf_counter() - start
    print(f"{duration:.3f}")


def test_optional_pin_dask_partial_construction_and_null_normalization():
    sd_client = create_dummy_client(workers=1, worker_threads=2, spawn_workers=2)
    set_seamless_dask_client(sd_client)
    try:

        @delayed
        def consume(a, x=None):
            if x is None:
                return a
            return a + x

        consume.optional_pins.add("x")

        absent = consume(10)
        submission = absent._build_dask_submission(
            sd_client,
            require_value=False,
            need_fat=False,
        )
        assert submission.optional_pins == frozenset({"x"})

        assert absent.run() == 10
        null_connected = consume(10, None)
        assert null_connected.run() == 10
        assert absent.transformation_checksum == null_connected.transformation_checksum

        non_null = consume(10, 5)
        assert non_null.run() == 15
        assert non_null.transformation_checksum != absent.transformation_checksum
    finally:
        set_seamless_dask_client(None)
        seamless.close()


def test_optional_pin_dask_dependency_failure_is_not_absence():
    sd_client = create_dummy_client(workers=1, worker_threads=2, spawn_workers=2)
    set_seamless_dask_client(sd_client)
    try:

        @delayed
        def boom():
            raise RuntimeError("dask upstream failed")

        @delayed
        def consume(a, x=None):
            return a

        consume.optional_pins.add("x")
        tf = consume(10, boom())

        with pytest.raises(Exception):
            tf.run()
        assert "dask upstream failed" in tf.exception
    finally:
        set_seamless_dask_client(None)
        seamless.close()
