import seamless.config

seamless.config.init()

from seamless.transformer import direct, delayed

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
