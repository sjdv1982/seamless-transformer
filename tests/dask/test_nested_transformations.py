import os
import pytest

import seamless.config

seamless.config.init()

from seamless.transformer import direct, delayed


def _canonical_result(value):
    """Convert transformation results to hashable structures."""

    if isinstance(value, dict):
        return tuple(sorted((k, _canonical_result(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_result(v) for v in value)
    return value


def test_nested_transformations_cached():
    main_pid = os.getpid()

    @direct
    def outer(label):
        import os
        from seamless.transformer import direct

        @direct
        def inner(label):
            import datetime
            import os
            import time

            return (
                label,
                os.getpid(),
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                time.perf_counter_ns(),
            )

        first = inner(label)
        second = inner(label)
        return {"outer_pid": os.getpid(), "results": (first, second)}

    first_run = outer("beta")
    first_result, second_result = first_run["results"]

    assert first_run["outer_pid"] != main_pid
    assert first_result == second_result
    assert first_result[1] != main_pid

    # inner_direct = inner("beta")  # does not work, need modules
    second_run = outer("beta")
    all_results = {
        _canonical_result(first_result),
        _canonical_result(second_result),
        # _canonical_result(inner_direct),
        *(_canonical_result(r) for r in second_run["results"]),
    }
    assert len(all_results) == 1
