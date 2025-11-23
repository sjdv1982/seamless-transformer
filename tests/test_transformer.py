import time

from seamless_transformer import transformer


def test_in_process_transformer_execution():
    """Simple smoke test mirroring the original Seamless direct test."""

    @transformer
    def func(a, b):
        return 10 * a + 2 * b

    assert func(30, 12) == 324
    assert func(40, 2) == 404

    @transformer
    def func2(a, b):
        import time

        time.sleep(2)
        return 8 * a - 3 * b

    start = time.perf_counter()
    result1 = func2(3, 12)
    first_duration = time.perf_counter() - start

    start = time.perf_counter()
    result2 = func2(3, 12)
    second_duration = time.perf_counter() - start

    print(first_duration)
    print(second_duration)

    assert result1 == result2 == -12
    assert first_duration >= 2
    assert second_duration < 0.5
