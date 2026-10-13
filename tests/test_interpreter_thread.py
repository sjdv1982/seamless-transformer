"""Language-independent owner-thread helper contract."""
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

import pytest

from seamless_transformer.languages import run_in_interpreter_thread


def test_many_callers_share_one_owner_thread(monkeypatch):
    monkeypatch.delenv("SEAMLESS_INTERPRETER_THREAD", raising=False)
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


def test_owner_exception_propagates_and_thread_remains_usable(monkeypatch):
    monkeypatch.delenv("SEAMLESS_INTERPRETER_THREAD", raising=False)
    name = "test-exception-" + uuid.uuid4().hex
    error = ValueError("interpreter failure")
    def fail():
        raise error
    with pytest.raises(ValueError, match="interpreter failure") as caught:
        run_in_interpreter_thread(name, fail)
    assert caught.value is error
    assert run_in_interpreter_thread(name, lambda: 42) == 42


def test_recursive_owner_call_runs_inline(monkeypatch):
    monkeypatch.delenv("SEAMLESS_INTERPRETER_THREAD", raising=False)
    name = "test-recursive-" + uuid.uuid4().hex
    def outer():
        outer_id = threading.get_ident()
        inner_id = run_in_interpreter_thread(name, threading.get_ident)
        return outer_id, inner_id
    with ThreadPoolExecutor(max_workers=1) as caller:
        outer_id, inner_id = caller.submit(run_in_interpreter_thread, name, outer).result(timeout=5)
    assert outer_id == inner_id


def test_environment_switch_runs_inline_on_each_caller(monkeypatch):
    name = "test-inline-" + uuid.uuid4().hex
    monkeypatch.delenv("SEAMLESS_INTERPRETER_THREAD", raising=False)
    original_owner = run_in_interpreter_thread(name, threading.get_ident)
    monkeypatch.setenv("SEAMLESS_INTERPRETER_THREAD", "0")
    def call():
        return threading.get_ident(), run_in_interpreter_thread(name, threading.get_ident)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda _: call(), range(12)))
    assert all(caller == actual and actual != original_owner for caller, actual in rows)
