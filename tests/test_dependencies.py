import pytest

from seamless.transformer import direct, delayed, worker


def _close_worker_manager():
    manager = worker._worker_manager
    if manager is not None:
        manager.close(wait=True)
    worker._worker_manager = None
    worker._set_has_spawned(False)


@pytest.fixture
def temporary_spawned_workers():
    """Spawn workers for the duration of a test and clean them up afterwards."""

    spawned_here = False
    if not worker.has_spawned():
        worker.spawn(3)
        spawned_here = True
    yield
    if spawned_here:
        _close_worker_manager()


def _test_dependencies(offset, is_parallel):
    import time

    @direct
    def slow_add(a, b, c=None, d=None, e=None):
        import time
        from seamless.transformer import global_lock

        with global_lock:
            time.sleep(1)
        return sum([x for x in (a, b, c, d, e) if x is not None])

    slow_add_delayed = slow_add.copy(return_transformation=True)

    @direct
    def fast_add(a, b, c=None, d=None, e=None):
        return sum([x for x in (a, b, c, d, e) if x is not None])

    assert fast_add(10, 20) == 30

    fast_add_delayed = fast_add.copy(return_transformation=True)
    print()

    start = time.perf_counter()
    result = fast_add(slow_add_delayed(2 + offset, 3), slow_add_delayed(4 + offset, 5))
    print(result)
    assert result == 14 + 2 * offset, result
    duration = time.perf_counter() - start
    print(f"{duration:.3f}")
    if is_parallel:
        assert duration > 1 and duration < 2
    else:
        assert duration > 2 and duration < 3

    start = time.perf_counter()
    result = fast_add(
        slow_add_delayed(2 + offset, 13), slow_add_delayed(4 + offset, 15)
    )
    assert result == 34 + 2 * offset, result
    duration = time.perf_counter() - start
    print(f"{duration:.3f}")
    if is_parallel:
        assert duration > 1 and duration < 2
    else:
        assert duration > 2 and duration < 3

    start = time.perf_counter()
    result = slow_add(slow_add_delayed(5 + offset, 6), fast_add_delayed(1 + offset, 2))
    assert result == 14 + 2 * offset, result
    duration = time.perf_counter() - start
    print(f"{duration:.3f}")
    assert duration > 2 and duration < 3

    if is_parallel:
        start = time.perf_counter()
        v1 = slow_add_delayed(9, -1, -2, -3)
        v2 = slow_add_delayed(10, -1, -2, -3)
        v3 = slow_add_delayed(11, -1, -2, -3)
        v4 = slow_add_delayed(12, -1, -2, -3)
        v5 = slow_add_delayed(13, -1, -2, -3)
        result = fast_add(v1, v2, v3, v4, v5)
        assert result == 3 + 4 + 5 + 6 + 7
        duration = time.perf_counter() - start
        print(f"{duration:.3f}")
        assert duration > 2 and duration < 3


def test_dependencies_in_process():
    if worker.has_spawned():
        _close_worker_manager()
    assert not worker.has_spawned()
    _test_dependencies(1000, is_parallel=False)


def test_dependencies_with_spawn(temporary_spawned_workers):
    _test_dependencies(2000, is_parallel=True)
