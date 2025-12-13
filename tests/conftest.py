import seamless
import seamless.shutdown as shutdown
import pytest
from seamless_transformer import worker
from multiprocessing import resource_tracker


@pytest.fixture(scope="session", autouse=True)
def _close_seamless_once():
    """Reset Seamless state and close once after the full test session."""

    seamless._closed = False  # allow tests to run even if a prior close occurred
    seamless._require_close = False
    shutdown._closed = False
    shutdown._closing = False
    worker._set_has_spawned(False)
    worker._worker_manager = None
    yield
    seamless.close()


@pytest.fixture(autouse=True)
def _reopen_seamless_between_tests():
    """Re-open Seamless if a previous test called seamless.close()."""

    seamless._closed = False
    seamless._require_close = False
    shutdown._closed = False
    shutdown._closing = False
    # Restart the multiprocessing resource tracker if a previous close shut it down.
    try:
        fd = getattr(resource_tracker._resource_tracker, "_fd", None)  # type: ignore[attr-defined]
    except Exception:
        fd = None
    if fd is None:
        try:
            resource_tracker._resource_tracker = None  # type: ignore[attr-defined]
            resource_tracker.ensure_running()
        except Exception:
            pass
    # do not touch worker manager here; individual tests manage spawning/cleanup
    yield
