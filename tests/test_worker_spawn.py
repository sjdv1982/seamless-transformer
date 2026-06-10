import concurrent.futures
import os

import pytest

from seamless import Checksum
from seamless.transformer import direct, delayed, has_spawned, spawn
from seamless_transformer import worker
from seamless_transformer.worker import _DaemonThreadPoolExecutor


@pytest.fixture(scope="session")
def spawned_workers():
    if not has_spawned():
        spawn(2)
    yield True
    # Do not call shutdown_workers() here; session fixture cleanup is handled elsewhere.


def test_spawned_workers_execute_transformations(spawned_workers):
    main_pid = os.getpid()

    @direct
    def pid_and_sum(a, b):
        import os

        return os.getpid(), a + b

    pid, value = pid_and_sum(2, 3)
    assert pid != main_pid
    assert value == 5


def test_nested_transformations_round_trip_to_parent(spawned_workers):
    main_pid = os.getpid()

    @direct
    def outer(x):
        from seamless.transformer import direct

        @direct
        def inner(x):
            import os

            return os.getpid(), x + 1

        pid_inner, intermediate = inner(x)
        import os

        return pid_inner, os.getpid(), intermediate + 1

    pid_inner, pid_outer, result = outer(10)
    assert result == 12
    assert pid_outer != main_pid
    assert pid_inner != main_pid


def test_spawn_is_singleton(spawned_workers):
    with pytest.raises(RuntimeError):
        spawn(1)


def test_daemon_thread_pool_executor_executes_tasks():
    executor = _DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="compat")
    try:
        future = executor.submit(lambda: 42)
        assert future.result(timeout=1) == 42
        threads = list(executor._threads)
        assert threads and all(thread.daemon for thread in threads)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def test_worker_manager_cancel_by_checksum_cancels_active_dispatches():
    manager = worker._WorkerManager.__new__(worker._WorkerManager)
    manager._active_dispatches = {}
    manager._active_handles = {}
    manager._cancelled_checksums = set()
    manager._handles = []
    manager._active_dispatch_lock = worker.threading.RLock()
    checksum = Checksum("6" * 64)
    future = concurrent.futures.Future()

    manager._remember_active_dispatch(checksum.hex(), future)

    assert manager.cancel_by_checksum(checksum) is True
    assert future.cancelled()
    assert manager._active_dispatches == {}
    # With no worker actually running the checksum, the cancellation guard is not
    # left dangling.
    assert checksum.hex() not in manager._cancelled_checksums
