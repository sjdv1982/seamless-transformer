"""Phase A: worker captures must remain local to their execution route."""

import asyncio
import io
import sys
import threading

import pytest

from seamless_transformer import worker


@pytest.fixture
def child_streams(monkeypatch):
    from seamless_config import extern_clients

    monkeypatch.setattr(worker, "set_is_worker", lambda value: None)
    monkeypatch.setattr(worker, "_patch_worker_primitives", lambda: None)
    monkeypatch.setattr(extern_clients, "set_remote_clients_from_env", lambda **kw: None)
    monkeypatch.setattr(worker, "_child_channel", None)
    monkeypatch.setattr(worker, "_child_loop", None)

    class Channel:
        ready_notified = True

        def add_request_handler(self, *args):
            pass

        def add_event_handler(self, *args):
            pass

    def initialize():
        # Pytest installs capture streams separately for setup and call. Install
        # the child router during call so the capture transition cannot replace it.
        stdout, stderr = io.StringIO(), io.StringIO()
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", stderr)
        # The initializer wraps Thread.start; register its original for teardown.
        monkeypatch.setattr(threading.Thread, "start", threading.Thread.start)
        asyncio.run(worker._child_initializer(Channel()))
        return stdout, stderr

    return initialize


def _wait(event):
    assert event.wait(5), "concurrent request did not reach its synchronization point"


def _launch(call):
    result, errors = [], []

    def run():
        try:
            result.append(call())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result, errors


def _join(task):
    thread, result, errors = task
    thread.join(8)
    assert not thread.is_alive(), "request deadlocked"
    assert not errors, errors
    return result[0]


def test_overlapping_requests_keep_streams_and_router_identity(child_streams, monkeypatch):
    child_streams = child_streams()
    a_entered, b_entered, a_exited = (threading.Event() for _ in range(3))
    original_streams = sys.stdout, sys.stderr

    def execute(payload, formatter):
        name = payload["name"]
        if name == "A":
            print("A-before")
            a_entered.set()
            _wait(b_entered)
            print("A-after")
            print("A-error", file=sys.stderr)
        else:
            b_entered.set()
            print("B-before")
            _wait(a_exited)
            print("B-after")
            print("B-error", file=sys.stderr)
        return name + " failed"

    monkeypatch.setattr(worker, "_execute_transformation_impl", execute)

    def run_a():
        try:
            return worker._execute_transformation_request({"name": "A"})
        finally:
            a_exited.set()

    a = _launch(run_a)
    _wait(a_entered)
    b = _launch(lambda: worker._execute_transformation_request({"name": "B"}))
    print("unrelated output")
    a_log, b_log = _join(a), _join(b)
    assert all(text in a_log for text in ("A-before", "A-after", "A-error"))
    assert all(text in b_log for text in ("B-before", "B-after", "B-error"))
    assert "B-before" not in a_log and "B-error" not in a_log
    assert "A-after" not in b_log and "A-error" not in b_log
    assert (sys.stdout, sys.stderr) == original_streams
    print("outside request")
    assert child_streams[0].getvalue() == "unrelated output\noutside request\n"
    assert child_streams[1].getvalue() == ""


def test_user_thread_inherits_route_and_late_output_uses_fallback(child_streams, monkeypatch):
    child_streams = child_streams()
    before, release = threading.Event(), threading.Event()
    children = []

    def execute(payload, formatter):
        def user_thread():
            print("inherited stdout")
            print("inherited stderr", file=sys.stderr)
            before.set()
            _wait(release)
            print("late stdout")
            print("late stderr", file=sys.stderr)

        child = _launch(user_thread)
        children.append(child)
        _wait(before)
        return "failed"

    monkeypatch.setattr(worker, "_execute_transformation_impl", execute)
    log = worker._execute_transformation_request({})
    release.set()
    # user_thread returns None, which the task helper still records.
    _join(children[0])
    assert "inherited stdout" in log and "inherited stderr" in log
    assert "late stdout" not in log and "late stderr" not in log
    assert child_streams[0].getvalue() == "late stdout\n"
    assert child_streams[1].getvalue() == "late stderr\n"


def test_nested_local_execution_has_its_own_capture(child_streams, monkeypatch):
    child_streams = child_streams()
    nested_logs = []

    def execute(payload, formatter):
        if payload.get("nested"):
            print("nested stdout")
            print("nested stderr", file=sys.stderr)
            return "nested failure"
        print("outer before")
        nested_logs.append(worker._execute_transformation({"nested": True}))
        print("outer after")
        return "outer failure"

    monkeypatch.setattr(worker, "_execute_transformation_impl", execute)
    outer = worker._execute_transformation_request({})
    assert "nested stdout" in nested_logs[0] and "nested stderr" in nested_logs[0]
    assert "nested stdout" not in outer and "nested stderr" not in outer
    assert "outer before" in outer and "outer after" in outer
    assert child_streams[0].getvalue() == child_streams[1].getvalue() == ""


def test_concurrent_tqdm_attribution_plain_neighbors_and_patch_lifetime(child_streams, monkeypatch):
    child_streams = child_streams()
    import tqdm
    from seamless_transformer import stream_tqdm

    original_tqdm = tqdm.tqdm
    original_install = stream_tqdm.install_tqdm_patch
    events = []
    owner = threading.local()
    a_entered, b_entered, a_exited, plain_done = (threading.Event() for _ in range(4))

    def install(notifier, **kwargs):
        name = owner.name
        return original_install(lambda event: events.append((name, dict(event))), **kwargs)

    monkeypatch.setattr(stream_tqdm, "install_tqdm_patch", install)

    def bar(desc):
        with tqdm.tqdm(total=1, desc=desc, mininterval=0) as progress:
            progress.update()

    def execute(payload, formatter):
        name = payload["name"]
        if name == "A":
            a_entered.set()
            _wait(b_entered)
            _wait(plain_done)
            bar("A bar")
        elif name == "B":
            b_entered.set()
            _wait(a_exited)
            bar("B after A")
        else:
            bar("plain request")
        return name + " failure"

    monkeypatch.setattr(worker, "_execute_transformation_impl", execute)

    def request(name, streaming=True):
        owner.name = name
        try:
            return worker._execute_transformation_request({"name": name, "streaming": streaming})
        finally:
            if name == "A":
                a_exited.set()

    a = _launch(lambda: request("A"))
    _wait(a_entered)
    b = _launch(lambda: request("B"))
    _wait(b_entered)
    # Main thread has no route even while two streaming requests are active.
    plain_output = io.StringIO()
    with tqdm.tqdm(total=1, desc="unrouted", file=plain_output, mininterval=0) as progress:
        progress.update()
    plain_log = request("plain", streaming=False)
    plain_done.set()
    _join(a)
    _join(b)
    opened = [(name, event["desc"]) for name, event in events if event["kind"] == "tqdm_open"]
    assert sorted(opened) == [("A", "A bar"), ("B", "B after A")]
    assert "unrouted" in plain_output.getvalue()
    assert "plain request" in plain_log
    assert tqdm.tqdm is original_tqdm
    for name in ("A", "B"):
        own = [event for event_owner, event in events if event_owner == name]
        assert any(event["kind"] == "tqdm_update" and event["n"] == 1 for event in own)
        assert any(event["kind"] == "tqdm_close" and event["n"] == 1 for event in own)


@pytest.mark.parametrize("replacement_lifetime", [False, True])
def test_tqdm_loader_finishing_after_patch_lifetime_cannot_restore_stale_alias(
    monkeypatch, replacement_lifetime
):
    import types
    from seamless_transformer import stream_tqdm

    # Restrict patching to a synthetic module so this loader scheduling probe
    # cannot mutate unrelated tqdm imports or create real progress bars.
    module = types.ModuleType("tqdm.std")

    class PlainTqdm:
        pass

    module.tqdm = PlainTqdm
    monkeypatch.setitem(sys.modules, "tqdm.std", module)
    monkeypatch.setattr(stream_tqdm, "_PATCHED_MODULES", ("tqdm.std",))
    first = stream_tqdm.install_tqdm_patch(lambda event: None)
    replacement = stream_tqdm.install_tqdm_patch(lambda event: None)
    first.__enter__()
    stale_wrapper = module.tqdm
    assert stale_wrapper is not PlainTqdm

    class Loader:
        def exec_module(self, loading_module):
            # The hierarchy loader temporarily exposes the original base class.
            assert module.tqdm is PlainTqdm
            first.__exit__(None, None, None)
            if replacement_lifetime:
                replacement.__enter__()
                assert module.tqdm is not stale_wrapper
                # Model an import still defining aliases after the replacement
                # lifetime started. The completed import must use that lifetime.
                module.tqdm = PlainTqdm

    try:
        stream_tqdm._PatchLoader(Loader(), "tqdm.auto").exec_module(
            types.ModuleType("tqdm.auto")
        )
        if replacement_lifetime:
            assert module.tqdm is not PlainTqdm
            assert module.tqdm is not stale_wrapper
            assert any(
                target is module and wrapper is module.tqdm
                for target, attr, wrapper, original in stream_tqdm._PATCHED_ORIGINALS
            )
        else:
            assert module.tqdm is PlainTqdm
    finally:
        first.__exit__(None, None, None)
        replacement.__exit__(None, None, None)
    assert module.tqdm is PlainTqdm
