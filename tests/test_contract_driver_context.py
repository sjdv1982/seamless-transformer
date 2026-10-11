"""Driver submission context is transported separately from child metadata."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _driver_launch_child(route: str, nonce: str):
    import asyncio
    from seamless.transformer import delayed, direct
    from seamless_transformer import worker
    from seamless_transformer.run import is_driver_context

    def child(nonce: str):
        from seamless.transformer import delayed
        from seamless_transformer import worker
        from seamless_transformer.run import is_driver_context

        def leaf(nonce: str):
            from seamless_transformer.run import is_driver_context
            return is_driver_context()

        records = []
        original_request = worker._request_parent_async

        async def capture_request(kind, payload):
            if kind == "delegate_transformation_submit":
                records.append({
                    "driver_context": payload.get("driver_context", False),
                    "driver": payload.get("tf_dunder", {}).get("__meta__", {}).get("driver", False),
                })
            return await original_request(kind, payload)

        worker._request_parent_async = capture_request
        try:
            leaf_builder = delayed(leaf)
            handle = leaf_builder(nonce)
            leaf_meta = dict(handle.meta)
            leaf_context = handle.run()
        finally:
            worker._request_parent_async = original_request
        return {
            "child_context": is_driver_context(),
            "leaf_context": leaf_context,
            "leaf_driver": leaf_meta.get("driver", False),
            "grandchild_payloads": records,
        }

    records = []
    original_request = worker._request_parent_async

    async def capture_request(kind, payload):
        if kind == "delegate_transformation_submit":
            records.append({
                "driver_context": payload.get("driver_context", False),
                "driver": payload.get("tf_dunder", {}).get("__meta__", {}).get("driver", False),
            })
        return await original_request(kind, payload)

    worker._request_parent_async = capture_request
    try:
        child_builder = direct(child) if route == "direct" else delayed(child)
        handle = child_builder.build(nonce)
        child_meta = dict(handle.meta)
        if route == "direct":
            result = child_builder(nonce)
        elif route == "run":
            result = handle.run()
        elif route == "start":
            result = handle.start().run()
        elif route == "computation":
            asyncio.run(handle.computation())
            result = handle.run()
        elif route == "task":
            async def launch():
                await handle.task()
            asyncio.run(launch())
            result = handle.run()
        else:
            raise AssertionError(route)
    finally:
        worker._request_parent_async = original_request
    result.update({
        "driver_body_context": is_driver_context(),
        "child_driver": child_meta.get("driver", False),
        "child_payloads": records,
    })
    return result


@pytest.mark.parametrize("mode", ["process", "spawn"])
@pytest.mark.parametrize("route", ["run", "start", "computation", "task", "direct"])
def test_driver_context_does_not_enter_child_dict_or_body(mode, route):
    script = '''
import json
import os
import uuid
import seamless
from seamless.transformer import delayed, spawn
from test_contract_driver_context import _driver_launch_child
mode = os.environ["DRIVER_CONTEXT_TEST_MODE"]
if mode == "spawn":
    spawn(2)
try:
    driver = delayed(_driver_launch_child)
    driver.driver = True
    result = driver(os.environ["DRIVER_CONTEXT_TEST_ROUTE"], uuid.uuid4().hex).run()
    assert result["driver_body_context"] is True, result
    assert result["child_driver"] is False, result
    assert result["leaf_driver"] is False, result
    assert result["child_context"] is False, result
    assert result["leaf_context"] is False, result
    if mode == "spawn":
        assert result["child_payloads"], result
        assert result["child_payloads"][0]["driver_context"] is True, result
        assert result["child_payloads"][0]["driver"] is False, result
        assert result["grandchild_payloads"], result
        assert all(p["driver_context"] is False for p in result["grandchild_payloads"]), result
        assert all(p["driver"] is False for p in result["grandchild_payloads"]), result
    print("DRIVER_OK")
finally:
    seamless.close()
'''
    env = os.environ.copy()
    env["DRIVER_CONTEXT_TEST_MODE"] = mode
    env["DRIVER_CONTEXT_TEST_ROUTE"] = route
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DRIVER_OK" in completed.stdout


def _fail_once(marker: str):
    from pathlib import Path
    path = Path(marker)
    if not path.exists():
        path.write_text("first attempt")
        raise RuntimeError("first attempt fails")
    return "retry succeeds"


def _report_body_driver_context(nonce: str):
    from seamless_transformer.run import is_driver_context
    return is_driver_context()


@pytest.mark.parametrize("first_context", [True, False])
def test_clear_exception_recaptures_retry_submission_context(monkeypatch, tmp_path, first_context):
    from seamless.transformer import delayed
    from seamless_transformer.run import _driver_context
    from seamless_transformer.transformation_cache import TransformationCache
    import seamless_transformer.transformation_cache as cache_module

    monkeypatch.setattr(cache_module, "get_execution", lambda: "process")
    captured = []
    original_run = TransformationCache.run

    async def capture_run(self, *args, **kwargs):
        captured.append(kwargs.get("driver_context", False))
        return await original_run(self, *args, **kwargs)

    monkeypatch.setattr(TransformationCache, "run", capture_run)
    handle = delayed(_fail_once)(str(tmp_path / "attempt"))
    with _driver_context(first_context):
        assert handle.compute() is None
    assert handle.exception is not None
    assert captured == [first_context]
    handle.clear_exception()
    with _driver_context(not first_context):
        assert handle.run() == "retry succeeds"
    assert captured == [first_context, not first_context]
    assert handle.meta.get("driver", False) is False


def test_task_preserves_submission_context_after_driver_scope_exits(monkeypatch):
    import asyncio
    import uuid
    from seamless.transformer import delayed
    from seamless_transformer.run import _driver_context, is_driver_context
    from seamless_transformer.transformation_cache import TransformationCache
    import seamless_transformer.transformation_cache as cache_module

    monkeypatch.setattr(cache_module, "get_execution", lambda: "process")
    captured = []
    original_run = TransformationCache.run

    async def capture_run(self, *args, **kwargs):
        captured.append(kwargs.get("driver_context", False))
        return await original_run(self, *args, **kwargs)

    monkeypatch.setattr(TransformationCache, "run", capture_run)

    async def submit_then_wait():
        handle = delayed(_report_body_driver_context)(uuid.uuid4().hex)
        with _driver_context(True):
            task = handle.task()
        assert is_driver_context() is False
        assert await task is False
        assert handle.meta.get("driver", False) is False

    asyncio.run(submit_then_wait())
    assert captured == [True]
