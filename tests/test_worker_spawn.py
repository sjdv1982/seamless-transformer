import os

import pytest

from seamless_transformer import transformer
from seamless_transformer.worker import has_spawned, spawn


@pytest.fixture(scope="session")
def spawned_workers():
    if not has_spawned():
        spawn(2)
    yield True
    # Do not call shutdown_workers() here; session fixture cleanup is handled elsewhere.


def test_spawned_workers_execute_transformations(spawned_workers):
    main_pid = os.getpid()

    @transformer
    def pid_and_sum(a, b):
        import os

        return os.getpid(), a + b

    pid, value = pid_and_sum(2, 3)
    assert pid != main_pid
    assert value == 5


def test_nested_transformations_round_trip_to_parent(spawned_workers):
    main_pid = os.getpid()

    @transformer
    def inner(x):
        import os

        return os.getpid(), x + 1

    @transformer
    def outer(x):
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
