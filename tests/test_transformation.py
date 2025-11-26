import time

from seamless_transformer import transformer

DELAY = 1.0


def test_transformation():
    """Test various forms of transformation execution"""

    @transformer
    def func(a, b, delay):
        import time

        print("RUN", a, b)
        time.sleep(delay)
        return 10 * a + 2 * b

    assert func(30, 12, 0) == 324
    assert func(40, 2, 0) == 404

    func2 = func.copy(return_transformation=True)
    start = time.perf_counter()

    result1 = func2(3, 12, DELAY).run()
    first_duration = time.perf_counter() - start

    start = time.perf_counter()
    result2 = func2(3, 12, DELAY).run()
    second_duration = time.perf_counter() - start

    print(first_duration)
    print(second_duration)

    assert result1 == result2 == 54
    assert first_duration >= DELAY
    assert second_duration < 0.5

    start = time.perf_counter()
    tfs = [func2(n, -1, DELAY) for n in range(4)]
    for tf in tfs:
        print(tf.run())
    duration1 = time.perf_counter() - start
    print(duration1)
    assert 4 * DELAY < duration1 < 4 * DELAY + 1, duration1

    start = time.perf_counter()
    tfs = [func2(n, -2, DELAY).start() for n in range(4)]
    for tf in tfs:
        print(tf.run())

    duration2 = time.perf_counter() - start
    print(duration2)
    assert 4 * DELAY < duration2 < 4 * DELAY + 1, duration2
