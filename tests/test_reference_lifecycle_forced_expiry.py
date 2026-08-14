from __future__ import annotations

import asyncio
import gc
import sys
from uuid import uuid4
from pathlib import Path

import pytest

from seamless import Buffer, CacheMissError, Checksum
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.reference_lifecycle import collect_refholder_claims
from seamless.transformer import delayed
from seamless_transformer.pretransformation import PreTransformation

sys.path.insert(0, str(Path(__file__).parent / "helpers"))
from reference_lifecycle import force_expiry  # noqa: E402


def _unique(label: str) -> str:
    return f"{label}-{uuid4().hex}"


def identity(value):
    return value


def suffix(value):
    return value + "-out"


def _guard_remote(monkeypatch):
    calls = {"buffer_remote": 0}
    try:
        import seamless_remote.buffer_remote as buffer_remote
    except ImportError:
        return calls

    async def missing(checksum):
        calls["buffer_remote"] += 1
        return None

    monkeypatch.setattr(buffer_remote, "get_buffer", missing)
    return calls


def _assert_miss(monkeypatch, checksum):
    calls = _guard_remote(monkeypatch)
    with pytest.raises((CacheMissError, RuntimeError, ValueError)):
        checksum.resolve()
    return calls


def _assert_claim(owner, checksum, role):
    claims = collect_refholder_claims([owner]).get(checksum, [])
    assert [(holder, claim_role) for holder, claim_role in claims] == [(owner, role)]


def _positive_then_negative(monkeypatch, owner, checksum, resolve):
    cache = get_buffer_cache()
    evidence = force_expiry(checksum, cache=cache)
    assert evidence.after["accounting"][0] >= 1
    assert evidence.after["accounting"][2] is True
    resolve()
    owner._release_refholds()
    gc.collect()
    force_expiry(checksum, cache=cache)
    _assert_miss(monkeypatch, checksum)


def test_standalone_builder_pin_survives_expiry_with_negative_control(monkeypatch):
    source = Buffer(_unique("builder-pin").encode())
    checksum = source.get_checksum()
    builder = delayed(identity)
    builder.args.value = checksum
    _assert_claim(builder, checksum, "pin:value")
    del source
    gc.collect()
    _positive_then_negative(monkeypatch, builder, checksum, lambda: checksum.resolve())


def test_checksum_backed_builder_code_survives_expiry(monkeypatch):
    code_buffer = Buffer(f"result = 1  # {_unique('code')}\n", "python")
    checksum = code_buffer.get_checksum()
    builder = delayed("result = value")
    builder.code = checksum
    _assert_claim(builder, checksum, "code")
    del code_buffer
    gc.collect()
    _positive_then_negative(monkeypatch, builder, checksum, lambda: checksum.resolve("python"))


def test_checksum_backed_builder_module_survives_expiry(monkeypatch):
    module_buffer = Buffer(_unique("module").encode(), "text")
    checksum = module_buffer.get_checksum()
    builder = delayed(identity)
    builder.modules.example = checksum
    _assert_claim(builder, checksum, "module:example")
    del module_buffer
    gc.collect()
    _positive_then_negative(monkeypatch, builder, checksum, lambda: checksum.resolve("text"))


def test_direct_transformation_input_is_adopted_and_protected(monkeypatch):
    input_buffer = Buffer(100000 + int(uuid4().hex[:6], 16), "int")
    checksum = input_buffer.get_checksum()
    transformation = delayed(identity)(checksum)
    transformation.compute()
    assert any(role == "input:value" for _, role in transformation._refheld_checksums())
    del input_buffer
    gc.collect()
    _positive_then_negative(monkeypatch, transformation, checksum, lambda: checksum.resolve("int"))


def test_evaluated_dependency_result_is_adopted_by_consumer(monkeypatch):
    producer = delayed(identity)(_unique("dependency"))
    producer._compute_dependency()
    result = producer._result_checksum_internal()
    assert result is not None
    consumer = delayed(identity)(producer)
    consumer.compute()
    assert any(
        checksum == result and role == "input:value"
        for checksum, role in consumer._refheld_checksums()
    )
    producer._release_refholds()
    _positive_then_negative(monkeypatch, consumer, result, lambda: result.resolve())
    consumer._release_refholds()


class _DelayedDependency:
    def __init__(self, checksum, delay, started):
        self._checksum = checksum
        self._delay = delay
        self.started = started
        self.exception = None

    async def _compute_dependency_async(self, *, require_value=False):
        self.started.set()
        if self._delay:
            await self._delay.wait()
        return self._checksum

    def _result_checksum_internal(self):
        return self._checksum


async def _run_adoption_case(bypass: bool):
    from seamless_transformer.transformation_class import Transformation

    fast_started = asyncio.Event()
    slow_started = asyncio.Event()
    slow_release = asyncio.Event()
    fast_buffer = Buffer(_unique("fast").encode(), "bytes")
    slow_buffer = Buffer(_unique("slow").encode(), "bytes")
    fast = fast_buffer.get_checksum()
    slow = slow_buffer.get_checksum()
    consumer = delayed(identity)(1)
    consumer._upstream_dependencies = {
        "fast": _DelayedDependency(fast, None, fast_started),
        "slow": _DelayedDependency(slow, slow_release, slow_started),
    }
    if bypass:
        consumer._adopt_input_checksum = lambda pin, checksum: None
    task = asyncio.create_task(consumer._run_dependencies_async(require_value=False))
    await asyncio.wait_for(fast_started.wait(), timeout=2)
    await asyncio.wait_for(slow_started.wait(), timeout=2)
    # The FIRST_COMPLETED loop has to adopt fast before slow is released.
    for _ in range(100):
        if not bypass and any(role == "input:fast" for _, role in consumer._refheld_checksums()):
            break
        await asyncio.sleep(0.01)
    return consumer, task, fast, slow_release, fast_buffer, slow_buffer


def test_fast_dependency_is_adopted_before_sibling_wait_completes(monkeypatch):
    async def run():
        consumer, task, fast, slow_release, fast_buffer, slow_buffer = await _run_adoption_case(False)
        del fast_buffer
        gc.collect()
        force_expiry(fast)
        assert fast.resolve("bytes").content.startswith(b"fast-")
        slow_release.set()
        await task
        consumer._release_refholds()
        del slow_buffer

    asyncio.run(run())


def test_fast_dependency_negative_control_fails_without_adoption(monkeypatch):
    async def run():
        consumer, task, fast, slow_release, fast_buffer, slow_buffer = await _run_adoption_case(True)
        del fast_buffer
        gc.collect()
        force_expiry(fast)
        calls = _assert_miss(monkeypatch, fast)
        assert calls["buffer_remote"] in (0, 1)
        slow_release.set()
        await task
        consumer._release_refholds()
        del slow_buffer

    asyncio.run(run())


def test_transformation_definition_is_retained_after_completion(monkeypatch):
    transformation = delayed(identity)(_unique("definition"))
    transformation.compute()
    definition = transformation._transformation_checksum_internal()
    assert definition is not None
    assert any(role == "definition" for _, role in transformation._refheld_checksums())
    _positive_then_negative(monkeypatch, transformation, definition, lambda: definition.resolve())


def test_public_transformation_result_holds_before_neutral_publication(monkeypatch):
    transformation = delayed(identity)(1)
    transformation._enable_result_holding()
    result_buffer = Buffer(_unique("public-result").encode(), "bytes")
    result = result_buffer.get_checksum()
    transformation._publish_result(result)
    assert get_buffer_cache().reference_snapshot()[result][0] == 1
    _positive_then_negative(monkeypatch, transformation, result, lambda: result.resolve("bytes"))


def test_neutral_result_publication_then_public_access_acquires_once(monkeypatch):
    transformation = delayed(identity)(1)
    result_buffer = Buffer(_unique("neutral-result").encode(), "bytes")
    result = result_buffer.get_checksum()
    transformation._publish_result(result)
    assert get_buffer_cache().reference_snapshot().get(result, (0, 0, False))[0] == 0
    transformation.buffer
    transformation._enable_result_holding()
    assert get_buffer_cache().reference_snapshot()[result][0] == 1
    _positive_then_negative(monkeypatch, transformation, result, lambda: result.resolve("bytes"))


def test_terminal_cancellation_releases_roles_and_blocks_late_publication(monkeypatch):
    input_buffer = Buffer(100000 + int(uuid4().hex[:6], 16), "int")
    checksum = input_buffer.get_checksum()
    transformation = delayed(identity)(checksum)
    del input_buffer
    gc.collect()
    transformation.construct()
    before = transformation._transformation_checksum_internal()
    assert before is not None
    assert transformation.cancel() is True
    assert not transformation._refheld_checksums()
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0
    late = Buffer(_unique("late-result").encode(), "bytes").get_checksum()
    assert transformation._publish_result(late) is None
    assert get_buffer_cache().reference_snapshot().get(late, (0, 0, False))[0] == 0
    force_expiry(checksum)
    _assert_miss(monkeypatch, checksum)


def test_cancellation_after_completed_publication_is_a_noop():
    transformation = delayed(identity)(_unique("completed"))
    result = transformation.compute()
    assert result is not None
    count = get_buffer_cache().reference_snapshot()[result][0]
    assert transformation.cancel() is False
    assert get_buffer_cache().reference_snapshot()[result][0] == count
    transformation._release_refholds()


def test_scratch_pretransformation_transfers_tempref_to_transformation_bridge():
    checksum = Buffer(_unique("scratch-input").encode(), "bytes").get_checksum()
    checksum.tempref(scratch=True)
    pre = PreTransformation(
        {
            "__language__": "python",
            "__output__": ("result", "bytes", None),
            "value": ("bytes", None, checksum),
        }
    )
    pre.prepare_transformation()
    assert pre._value_refs == []
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0
    from seamless_transformer.transformation_class import transformation_from_pretransformation

    transformation = transformation_from_pretransformation(
        pre,
        upstream_dependencies={},
        meta={},
        scratch=False,
    )
    assert any(role == "input:value" for _, role in transformation._refheld_checksums())
    assert get_buffer_cache().reference_snapshot()[checksum][2] is True
    transformation._release_refholds()


def test_requested_scratch_result_is_protected_but_unrequested_result_is_purgeable():
    unrequested_builder = delayed(suffix)
    unrequested_builder.scratch = True
    unrequested = unrequested_builder(_unique("unrequested"))
    unrequested_result = unrequested._compute_dependency()
    assert unrequested_result is not None
    assert get_buffer_cache().reference_snapshot().get(unrequested_result, (0, 0, False))[0] == 0
    assert get_buffer_cache().purge_scratch(unrequested_result) == 1

    requested_builder = delayed(suffix)
    requested_builder.scratch = True
    requested = requested_builder(_unique("requested"))
    requested_result = requested.compute()
    assert requested_result is not None
    assert get_buffer_cache().reference_snapshot()[requested_result][0] == 1
    assert get_buffer_cache().purge_scratch(requested_result) == 0
    requested._release_refholds()
