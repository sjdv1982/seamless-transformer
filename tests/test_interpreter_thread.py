"""Language-independent owner-thread helper contract."""
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

import pytest

from seamless_transformer.languages import run_in_interpreter_thread


def test_many_callers_share_one_owner_thread():
    name = "test-owner-" + uuid.uuid4().hex
    def call(index):
        caller = threading.get_ident()
        owner, result = run_in_interpreter_thread(name, lambda value: (threading.get_ident(), value), index)
        return caller, owner, result
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(call, range(32)))
    assert len({row[1] for row in rows}) == 1
    assert all(caller != owner for caller, owner, _ in rows)
    assert [row[2] for row in rows] == list(range(32))


def test_owner_exception_propagates_and_thread_remains_usable():
    name = "test-exception-" + uuid.uuid4().hex
    error = ValueError("interpreter failure")
    def fail():
        raise error
    with pytest.raises(ValueError, match="interpreter failure") as caught:
        run_in_interpreter_thread(name, fail)
    assert caught.value is error
    assert run_in_interpreter_thread(name, lambda: 42) == 42


def test_recursive_owner_call_runs_inline():
    name = "test-recursive-" + uuid.uuid4().hex
    def outer():
        outer_id = threading.get_ident()
        inner_id = run_in_interpreter_thread(name, threading.get_ident)
        return outer_id, inner_id
    with ThreadPoolExecutor(max_workers=1) as caller:
        outer_id, inner_id = caller.submit(run_in_interpreter_thread, name, outer).result(timeout=5)
    assert outer_id == inner_id

