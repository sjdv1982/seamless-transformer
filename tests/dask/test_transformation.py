import time

import seamless.config

seamless.config.init()

from seamless.transformer import direct, delayed

DELAY = 1.0


def test_transformation():
    """Test various forms of transformation execution"""

    @direct
    def func(a, b, delay) -> float:
        import time
        from seamless.transformer import global_lock

        with global_lock:
            print("RUN", a, b)
            time.sleep(delay)
        return float(10 * a + 2 * b)

    assert func(30, 12, 0) == 324
    assert func(40, 2, 0) == 404
    x = func(1, 2, 3)
    assert x == 14

    func2 = delayed(func)

    start = time.perf_counter()
    tf1 = func2(3, 12, DELAY)
    tf1.construct()
    result1 = tf1.run()
    first_duration = time.perf_counter() - start

    start = time.perf_counter()
    tf2 = func2(3, 12, DELAY)
    result2 = tf2.run()
    second_duration = time.perf_counter() - start

    print(first_duration)
    print(second_duration)
    print(tf2.transformation_checksum, tf2.transformation_checksum)
    assert tf1.transformation_checksum == tf2.transformation_checksum

    assert result1 == result2 == 54
    if first_duration > 0.4:  # cache miss
        assert first_duration >= DELAY
    assert second_duration < 0.5

    start = time.perf_counter()
    tfs = [func2(n, -1, DELAY) for n in range(4)]
    for tf in tfs:
        y = tf.run()
        print(y)
    duration1 = time.perf_counter() - start
    print(duration1)
    if duration1 > 0.4:  # cache miss:
        assert 4 * DELAY < duration1 < 4 * DELAY + 1, duration1

    start = time.perf_counter()
    tfs = [func2(n, -2, DELAY).start() for n in range(4)]
    for tf in tfs:
        print(tf.run())

    duration2 = time.perf_counter() - start
    print(duration2)
    if duration1 > 0.4 and duration2 > 0.4:  # cache miss:
        # In Dask mode the started tasks should complete faster than the sequential run.
        assert (
            duration2 < duration1
        ), f"Dask start/run was not faster ({duration2}s >= {duration1}s)"
