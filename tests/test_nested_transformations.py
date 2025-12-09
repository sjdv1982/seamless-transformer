import os
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
        worker.spawn(2)
        spawned_here = True
    yield
    if spawned_here:
        _close_worker_manager()


def _canonical_result(value):
    """Convert transformation results to hashable structures."""

    if isinstance(value, dict):
        return tuple(sorted((k, _canonical_result(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_result(v) for v in value)
    return value


def test_nested_transformations_cached_in_process():
    if worker.has_spawned():
        _close_worker_manager()
    assert not worker.has_spawned()

    @direct
    def inner(label):
        import datetime
        import time
        import os

        return (
            label,
            os.getpid(),
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            time.perf_counter_ns(),
        )

    @direct
    def outer(label):
        first = inner(label)
        second = inner(label)
        return first, second

    first, second = outer("alpha")
    inner_direct = inner("alpha")
    repeat_first, repeat_second = outer("alpha")

    unique_results = {
        _canonical_result(first),
        _canonical_result(second),
        _canonical_result(inner_direct),
        _canonical_result(repeat_first),
        _canonical_result(repeat_second),
    }
    assert len(unique_results) == 1


def test_nested_transformations_cached_with_spawn(temporary_spawned_workers):
    main_pid = os.getpid()

    @direct
    def inner(label):
        import datetime
        import time
        import os

        return (
            label,
            os.getpid(),
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            time.perf_counter_ns(),
        )

    @direct
    def outer(label):
        import os

        first = inner(label)
        second = inner(label)
        return {"outer_pid": os.getpid(), "results": (first, second)}

    first_run = outer("beta")
    first_result, second_result = first_run["results"]

    assert first_run["outer_pid"] != main_pid
    assert first_result == second_result
    assert first_result[1] != main_pid

    inner_direct = inner("beta")
    second_run = outer("beta")
    all_results = {
        _canonical_result(first_result),
        _canonical_result(second_result),
        _canonical_result(inner_direct),
        *(_canonical_result(r) for r in second_run["results"]),
    }
    assert len(all_results) == 1
