import signal
from threading import Event

import pytest
from seamless import Checksum

from seamless_transformer.api import run_transformation
from seamless_transformer.cmd import bash_transformation
from seamless_transformer.cmd.api import main as run_main


class _FakeCache:
    def __init__(self):
        self.canceled = []

    def cancel_by_checksum(self, checksum):
        self.canceled.append(Checksum(checksum).hex())
        return True


def _exercise_sigint_context(monkeypatch, module, context_factory):
    checksum = Checksum("9" * 64)
    fake_cache = _FakeCache()
    handlers = {}

    monkeypatch.setattr(module, "get_transformation_cache", lambda: fake_cache)
    monkeypatch.setattr(module.signal, "getsignal", lambda signum: None)

    def fake_signal(signum, handler):
        handlers[signum] = handler

    monkeypatch.setattr(module.signal, "signal", fake_signal)

    with pytest.raises(KeyboardInterrupt):
        with context_factory(checksum):
            handlers[signal.SIGINT](signal.SIGINT, None)

    assert fake_cache.canceled == [checksum.hex()]


def test_run_transformation_sigint_cancels_current_checksum(monkeypatch):
    _exercise_sigint_context(
        monkeypatch,
        run_transformation,
        run_transformation._cancel_current_on_termination,
    )


def test_seamless_run_sigint_cancels_current_checksum(monkeypatch):
    _exercise_sigint_context(
        monkeypatch,
        run_main,
        run_main._cancel_current_on_termination,
    )


def test_seamless_run_sigint_cancels_active_dask_submission(monkeypatch):
    checksum = Checksum("a" * 64)
    release = Event()
    calls = []

    class _FakeThinFuture:
        def result(self):
            handlers[signal.SIGINT](signal.SIGINT, None)
            release.wait(5)
            return checksum.hex(), "b" * 64, None

        def done(self):
            return False

        def cancelled(self):
            return False

    class _FakeBaseFuture:
        def add_done_callback(self, _callback):
            return None

        def done(self):
            return False

        def cancelled(self):
            return False

    class _FakeFutures:
        base = _FakeBaseFuture()
        thin = _FakeThinFuture()
        fat = None
        tf_checksum = checksum.hex()
        result_checksum = None

    class _FakeDaskClient:
        def _cached_transformation_for_submission(self, _submission):
            return None

        def get_fat_checksum_future(self, _checksum):
            return object()

        def submit_transformation(self, _submission, *, need_fat=False):
            del need_fat
            return _FakeFutures()

        def cancel_by_checksum(self, tf_checksum):
            calls.append(Checksum(tf_checksum).hex())
            release.set()
            return True

    handlers = {}
    monkeypatch.setattr(
        "seamless_dask.transformer_client.get_seamless_dask_client",
        lambda: _FakeDaskClient(),
    )
    monkeypatch.setattr(run_main.signal, "getsignal", lambda signum: None)
    monkeypatch.setattr(
        run_main.signal,
        "signal",
        lambda signum, handler: handlers.setdefault(signum, handler),
    )

    transformation_dict = {
        "__language__": "bash",
        "__output__": ("result", "bytes", None),
        "code": ("text", None, "c" * 64),
    }

    with pytest.raises(KeyboardInterrupt):
        with run_main._cancel_current_on_termination(checksum):
            bash_transformation.run_transformation(
                transformation_dict,
                undo=False,
                fingertip=False,
                scratch=False,
            )

    assert calls == [checksum.hex()]
