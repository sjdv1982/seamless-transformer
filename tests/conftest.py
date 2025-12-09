import seamless
import seamless.shutdown as shutdown
import pytest
from seamless.transformer import worker


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
