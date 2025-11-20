import time
import threading
import warnings

from seamless_transformer import transformer


def test_transformer_execution(recwarn):
    """Simple smoke test mirroring the original Seamless direct test."""

    @transformer
    def func(a, b):
        return 10 * a + 2 * b

    assert func(30, 12) == 324
    assert func(40, 2) == 404

    @transformer
    def func2(a, b):
        import time

        time.sleep(2)
        return 8 * a - 3 * b

    start = time.perf_counter()
    result1 = func2(3, 12)
    first_duration = time.perf_counter() - start

    start = time.perf_counter()
    result2 = func2(3, 12)
    second_duration = time.perf_counter() - start

    print(first_duration)
    print(second_duration)

    assert result1 == result2 == -12
    assert first_duration >= 2
    assert second_duration < 0.5
    _assert_no_multithread_warning(recwarn)


def _assert_no_multithread_warning(recwarn):
    warned = [
        w
        for w in recwarn
        if issubclass(w.category, DeprecationWarning)
        and "multi-threaded" in str(w.message)
    ]
    assert not warned


def test_fork_warning_with_foreign_thread(recwarn):
    stop_event = threading.Event()

    def run_forever():
        while not stop_event.is_set():
            time.sleep(0.05)

    thread = threading.Thread(target=run_forever)
    thread.start()
    from seamless_transformer import run as transformer_run

    old_flag = transformer_run._BUFFER_WRITER_HOOK_ACTIVE
    transformer_run._BUFFER_WRITER_HOOK_ACTIVE = False
    try:
        try:
            import multiprocessing.popen_fork as _popen_fork
        except ImportError:  # pragma: no cover
            _popen_fork = None
        if _popen_fork is not None:
            registry = getattr(_popen_fork, "__warningregistry__", None)
            if registry is not None:
                registry.clear()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)

            @transformer
            def func():
                return 1

            assert func() == 1
        warnings_received = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "multi-threaded" in str(w.message)
        ]
        assert warnings_received, "Expected multi-threaded DeprecationWarning"
    finally:
        transformer_run._BUFFER_WRITER_HOOK_ACTIVE = old_flag
        stop_event.set()
        thread.join()


def test_foreign_thread_warning_with_buffer_writer(recwarn):
    """Ensure buffer_writer hook emits warning when extra threads exist."""
    from seamless.caching import buffer_writer

    stop_event = threading.Event()

    def run_forever():
        while not stop_event.is_set():
            time.sleep(0.05)

    thread = threading.Thread(target=run_forever)
    thread.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)

            buffer_writer._before_fork()
        warnings_received = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "multi-threaded" in str(w.message)
        ]
        assert warnings_received, "Expected warning from buffer_writer hook"
    finally:
        stop_event.set()
        thread.join()
