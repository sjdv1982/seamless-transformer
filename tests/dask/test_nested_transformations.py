import os
import tempfile
import time
from pathlib import Path
import pytest

import seamless.config

from seamless.transformer import direct, delayed
from seamless_dask.transformer_client import get_seamless_dask_client


def _canonical_result(value):
    """Convert transformation results to hashable structures."""

    if isinstance(value, dict):
        return tuple(sorted((k, _canonical_result(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_result(v) for v in value)
    return value


_LABEL_COUNTER_PATH = Path(tempfile.gettempdir()) / "seamless-test-label-counter"


def _next_label() -> str:
    """Persist label increments across test runs to avoid cache-only paths."""

    try:
        value = int(_LABEL_COUNTER_PATH.read_text().strip())
    except Exception:
        value = 7
    label = f"gamma{value}"
    try:
        _LABEL_COUNTER_PATH.write_text(str(value + 1))
    except Exception:
        pass
    return label


def test_nested_transformations_cached():

    seamless.config.init()
    label = _next_label()
    main_pid = os.getpid()
    sd_client = get_seamless_dask_client()
    assert sd_client is not None
    dask_client = sd_client.client
    # Prime task stream recording so the timing window captures tasks.
    try:
        dask_client.get_task_stream()
    except Exception:
        pass
    start_ts = time.time()

    @direct
    def outer(label):
        import os
        from seamless.transformer import direct

        import time

        time.sleep(1.5)

        @direct
        def inner(label):

            import datetime
            import os
            import time

            time.sleep(3)

            return (
                label,
                os.getpid(),
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                time.perf_counter_ns(),
            )

        first = inner(label)
        second = inner(label)
        third = inner(label + "-x")
        fourth = inner(label + "-x")

        return {"outer_pid": os.getpid(), "results": (first, second, third, fourth)}

    first_run = outer(label)
    first_result, second_result, third_result, fourth_result = first_run["results"]

    assert first_run["outer_pid"] != main_pid
    assert first_result == second_result
    assert first_result[1] != main_pid
    assert third_result == fourth_result

    # inner_direct = i_resultnner(LABEL)  # does not work, need modules
    second_run = outer(label)
    stop_ts = time.time()
    all_results = {
        _canonical_result(first_result),
        _canonical_result(second_result),
        _canonical_result(third_result),
        _canonical_result(fourth_result),
        # _canonical_result(inner_direct),
        *(_canonical_result(r) for r in second_run["results"]),
    }
    assert len(all_results) == 2
    print(all_results)

    task_stream = dask_client.get_task_stream(start=start_ts, stop=stop_ts)
    base_keys = {
        entry.get("key")
        for entry in task_stream
        if str(entry.get("key", "")).startswith("base-")
    }
    assert len(base_keys) == 3, base_keys
