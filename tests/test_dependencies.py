import os
import pytest

from seamless_transformer import transformer, worker


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
        worker.spawn(2)
        spawned_here = True
    yield
    if spawned_here:
        _close_worker_manager()


def _test_dependencies(offset):
    import time

    @transformer
    def slow_add(a, b):
        import time

        time.sleep(1)
        return a + b

    slow_add_delayed = slow_add.copy(return_transformation=True)

    @transformer
    def fast_add(a, b):
        return a + b

    fast_add_delayed = fast_add.copy(return_transformation=True)

    start = time.perf_counter()
    result = fast_add(slow_add_delayed(2 + offset, 3), slow_add_delayed(4 + offset, 5))
    assert result == 14 + 2 * offset, result
    print(f"{time.perf_counter() - start:.3f}")

    result = slow_add(slow_add_delayed(5 + offset, 6), fast_add_delayed(1 + offset, 2))
    assert result == 14 + 2 * offset, result
    print(f"{time.perf_counter() - start:.3f}")


def test_dependencies_in_process():
    if worker.has_spawned():
        _close_worker_manager()
    assert not worker.has_spawned()
    _test_dependencies(1000)


def test_dependencies_with_spawn(temporary_spawned_workers):
    _test_dependencies(2000)
