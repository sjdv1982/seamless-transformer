"""Holder lifetime, isolation and cancellation of the process lock."""

import contextvars
import importlib
import threading
import time

import pytest

from seamless_transformer import process_lock
from seamless_transformer.transformation_cache import TransformationCancelledError


def test_scope_false_hides_outer_holder_without_dropping_it(monkeypatch):
    monkeypatch.delenv(process_lock.NOTIFY_ENV, raising=False)
    waiting = threading.Event()
    acquired = threading.Event()

    def contender():
        waiting.set()
        with process_lock.scope(True):
            acquired.set()

    with process_lock.scope(True):
        holder = process_lock.current_holder()
        assert holder is not None
        thread = threading.Thread(target=contender)
        thread.start()
        assert waiting.wait(2)
        with process_lock.scope(False):
            assert process_lock.current_holder() is None
            process_lock.drop_for_nested_work()
        assert process_lock.current_holder() is holder
        assert not acquired.wait(.05), "scope(False) released the outer holder"
    assert acquired.wait(2)
    thread.join(2)
    assert not thread.is_alive()
    assert process_lock.current_holder() is None


def test_drop_is_idempotent_and_never_releases_another_holder():
    acquired = threading.Event()
    release = threading.Event()
    second_acquired = threading.Event()

    def contender():
        with process_lock.scope(True):
            acquired.set()
            assert release.wait(3)

    def second_contender():
        with process_lock.scope(True):
            second_acquired.set()

    with process_lock.scope(True):
        context = contextvars.copy_context()
        first = threading.Thread(target=contender)
        first.start()
        process_lock.drop_for_nested_work()
        assert acquired.wait(2)
        process_lock.drop_for_nested_work()
        second = threading.Thread(target=second_contender)
        second.start()
        assert not second_acquired.wait(.05)
    # A stale copied context must not release a new holder after its body ended.
    context.run(process_lock.drop_for_nested_work)
    assert not second_acquired.wait(.05)
    release.set()
    assert second_acquired.wait(2)
    first.join(2)
    second.join(2)
    assert not first.is_alive() and not second.is_alive()


def test_cancel_waiting_scope_and_continue_after_release():
    entered = threading.Event()
    canceled = threading.Event()
    outcomes = []

    def waiter():
        entered.set()
        try:
            with process_lock.scope(True, should_abort=canceled.is_set):
                outcomes.append("body ran")
        except TransformationCancelledError:
            outcomes.append("cancelled")

    with process_lock.scope(True):
        thread = threading.Thread(target=waiter)
        thread.start()
        assert entered.wait(2)
        canceled.set()
        thread.join(2)
        assert not thread.is_alive()
        assert outcomes == ["cancelled"]
    with process_lock.scope(True):
        assert process_lock.current_holder() is not None


@pytest.mark.parametrize("module_name", ["seamless_transformer", "seamless.transformer"])
def test_retired_global_lock_attribute_points_to_metadata(module_name):
    module = importlib.import_module(module_name)
    assert "global_lock" not in module.__all__
    with pytest.raises(AttributeError, match="process_lock"):
        getattr(module, "global_lock")
    with pytest.raises(ImportError):
        exec(f"from {module_name} import global_lock", {})


def _forbidden_body(marker: str):
    from pathlib import Path
    Path(marker).touch()
    return 1


def test_executor_rejects_conflicting_merged_metadata_before_body(tmp_path):
    from seamless.transformer import delayed
    from seamless_transformer.run import run_transformation_dict
    builder = delayed(_forbidden_body)
    marker = tmp_path / "body"
    checksum = builder(str(marker)).construct()
    transformation_dict = checksum.resolve().get_value("plain")
    # Exercise merged metadata, including an invalid pair assembled across the two sources.
    transformation_dict["__meta__"] = {"process_lock": True}
    with pytest.raises(Exception) as exc:
        run_transformation_dict(transformation_dict, checksum, tf_dunder={"__meta__": {"driver": True}})
    assert "driver" in str(exc.value) and "process_lock" in str(exc.value)
    assert not marker.exists()


def test_cancel_detected_immediately_after_acquire_releases_lock():
    calls = []

    def should_abort():
        calls.append(None)
        return len(calls) == 2

    with pytest.raises(TransformationCancelledError):
        with process_lock.scope(True, should_abort=should_abort):
            pytest.fail("cancellation raced acquisition but the body started")
    assert len(calls) == 2
    deadline = time.monotonic() + 1
    with process_lock.scope(True, should_abort=lambda: time.monotonic() > deadline):
        assert process_lock.current_holder().held
