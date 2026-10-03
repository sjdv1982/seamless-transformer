import asyncio

import pytest

from seamless import CacheMissError, Checksum, Expression
from seamless.error_envelope import ExecutionCanceledError
from aiohttp import ClientConnectionError, ClientPayloadError
import seamless.config
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.checksum.cached_calculate_checksum import checksum_cache
import seamless.checksum.expression as expression_mod
from seamless.transformer import delayed

from seamless.config import set_stage

set_stage("fingertip")


def _drop_expression_buffer(checksum):
    checksum = Checksum(checksum)
    cache = get_buffer_cache()
    with cache.lock:
        cache.weak_cache.pop(checksum, None)
        cache.strong_cache.pop(checksum, None)
    checksum_cache.pop(checksum, None)
    expression_mod._expression_result_buffers.pop(checksum, None)


def test_fingertip_recompute_scratch():
    seamless.config.init()

    @delayed
    def func(a, b) -> float:
        return 2.13 * a + 2.86 * b

    func.scratch = True
    func.local = True

    tf = func(2, 3)
    print("Transformation:", tf.construct())
    result_checksum = tf.compute()
    assert isinstance(result_checksum, Checksum), tf.exception
    print("Result:", result_checksum)

    result_checksum.tempref()
    get_buffer_cache().purge_scratch(result_checksum)

    with pytest.raises(CacheMissError):
        result_checksum.resolve()

    value = asyncio.run(result_checksum.fingertip("mixed"))
    assert value == pytest.approx(12.84)

    buf = result_checksum.resolve()
    assert buf.get_value("mixed") == pytest.approx(12.84)

    tf._release_refholds()
    get_buffer_cache().purge_scratch(result_checksum)
    with pytest.raises(CacheMissError):
        result_checksum.resolve()

    value2 = asyncio.run(result_checksum.fingertip("mixed"))
    assert value2 == pytest.approx(12.84)

    seamless.close()


def test_fingertip_recovers_expression_over_transformation_result():
    seamless.config.init()
    expression_mod.get_expression_cache().clear()

    @delayed
    def produce() -> dict:
        return {"a": "via-transform"}

    produce.scratch = True
    produce.local = True
    produce.celltypes.result = "plain"

    tf = produce()
    tf_result = tf.compute()
    assert asyncio.run(tf_result.fingertip("plain")) == {"a": "via-transform"}
    expression = Expression(tf_result, "a", input_celltype="plain", celltype="str")
    expression_result = expression.compute()

    tf_result.tempref()
    get_buffer_cache().purge_scratch(tf_result)
    _drop_expression_buffer(expression_result)

    value = asyncio.run(expression_result.fingertip("str"))

    assert value == "via-transform"
    assert expression_result.resolve("str") == "via-transform"

    seamless.close()


def test_run_fingertip_scratch():
    seamless.config.init()

    @delayed
    def func(a, b) -> float:
        return 2.13 * a + 2.86 * b

    func.scratch = True
    func.local = True

    tf = func(2, 3)
    value = tf.run()
    assert value == pytest.approx(12.84)

    seamless.close()


def _random_scratch_transformation():
    import uuid
    @delayed
    def random_result(nonce) -> bytes:
        import os
        return os.urandom(32)
    random_result.scratch = True
    random_result.local = True
    tf = random_result(uuid.uuid4().hex)
    result = tf.compute()
    assert isinstance(result, Checksum), tf.exception
    return tf, result


def test_irreproducible_fingertip_preserves_recorded_identity():
    from seamless import FingertipCategory
    from seamless_remote import database_remote
    from seamless_transformer.transformation_cache import get_transformation_cache
    seamless.config.init()
    tf, recorded = _random_scratch_transformation()
    tf_checksum = tf.construct()
    get_buffer_cache().purge_scratch(recorded)
    with pytest.raises(CacheMissError) as caught:
        asyncio.run(recorded.fingertip())
    assert caught.value.fingertip_category == FingertipCategory.IRREPRODUCIBLE_TRANSFORMATION
    assert asyncio.run(database_remote.get_transformation_result(tf_checksum)) == recorded
    assert tf_checksum in asyncio.run(database_remote.get_rev_transformations(recorded))
    rows = asyncio.run(database_remote.get_irreproducible_records(tf_checksum))
    assert len(rows) == 1 and rows[0]["result"] != recorded.hex()
    assert get_transformation_cache()._transformation_cache[tf_checksum] == recorded


def test_plain_rerun_returns_divergence_unrecorded():
    from seamless_remote import database_remote
    from seamless_transformer.transformation_cache import get_transformation_cache
    seamless.config.init()
    tf, recorded = _random_scratch_transformation()
    tf_checksum = tf.construct()
    get_buffer_cache().purge_scratch(recorded)
    produced = asyncio.run(get_transformation_cache().run(
        tf_checksum.resolve("plain"), tf_checksum=tf_checksum,
        tf_dunder=tf._tf_dunder, scratch=True, require_value=True,
    ))
    assert produced != recorded
    assert asyncio.run(database_remote.get_transformation_result(tf_checksum)) == recorded
    assert get_transformation_cache()._transformation_cache[tf_checksum] == recorded
    rows = asyncio.run(database_remote.get_irreproducible_records(tf_checksum))
    assert len(rows) == 1 and rows[0]["result"] == produced.hex()


def test_reproducible_rerun_writes_nothing(monkeypatch):
    import uuid
    from seamless_remote import database_remote
    seamless.config.init()
    @delayed
    def reproduce(nonce):
        return nonce + "-reproduced"
    reproduce.scratch = True
    reproduce.local = True
    tf = reproduce(uuid.uuid4().hex)
    result = tf.compute()
    writes = []
    async def unexpected_write(*args):
        writes.append(args)
        raise AssertionError("recomputation wrote an existing mapping")
    monkeypatch.setattr(database_remote, "set_transformation_result", unexpected_write)
    monkeypatch.setattr(database_remote, "set_execution_record", unexpected_write)
    get_buffer_cache().purge_scratch(result)
    assert asyncio.run(result.fingertip("mixed")).endswith("-reproduced")
    assert writes == []


def test_failed_transformation_category(monkeypatch):
    import uuid
    from seamless import FingertipCategory
    seamless.config.init()
    variable = "SEAMLESS_FINGERTIP_TEST_RUN"
    monkeypatch.setenv(variable, "first")
    @delayed
    def first_only(nonce):
        import os
        if "SEAMLESS_FINGERTIP_TEST_RUN" not in os.environ:
            raise RuntimeError("recompute failure")
        return nonce + "-first-only"
    first_only.scratch = True
    first_only.local = True
    tf = first_only(uuid.uuid4().hex)
    result = tf.compute()
    monkeypatch.delenv(variable)
    get_buffer_cache().purge_scratch(result)
    with pytest.raises(CacheMissError) as caught:
        asyncio.run(result.fingertip())
    assert caught.value.fingertip_category == FingertipCategory.FAILED_TRANSFORMATION


def test_materialization_without_candidate_or_input(monkeypatch):
    import uuid
    from seamless import FingertipCategory, Buffer
    seamless.config.init()
    with pytest.raises(CacheMissError) as caught:
        asyncio.run(Checksum("d" * 64).fingertip())
    assert caught.value.fingertip_category == FingertipCategory.MATERIALIZATION
    @delayed
    def consume(value):
        return value + "-consumed"
    consume.scratch = True
    consume.local = True
    source = Buffer(uuid.uuid4().hex, "mixed")
    tf = consume(source.get_checksum())
    result = tf.compute()
    assert isinstance(result, Checksum), tf.exception
    input_checksum = source.get_checksum()
    original_resolve = Checksum.resolve
    original_resolution = Checksum.resolution
    def resolve(checksum, *args, **kwargs):
        if checksum == input_checksum:
            raise CacheMissError(checksum)
        return original_resolve(checksum, *args, **kwargs)
    async def resolution(checksum, *args, **kwargs):
        if checksum == input_checksum:
            raise CacheMissError(checksum)
        return await original_resolution(checksum, *args, **kwargs)
    monkeypatch.setattr(Checksum, "resolve", resolve)
    monkeypatch.setattr(Checksum, "resolution", resolution)
    _drop_expression_buffer(input_checksum)
    get_buffer_cache().purge_scratch(result)
    with pytest.raises(CacheMissError) as caught:
        asyncio.run(result.fingertip())
    assert caught.value.fingertip_category == FingertipCategory.MATERIALIZATION


@pytest.mark.parametrize("reverse", [False, True])
def test_candidate_ranking_is_order_independent(monkeypatch, reverse):
    from seamless import FingertipCategory
    import seamless_transformer.transformation_cache as cache_mod
    from seamless_remote import database_remote
    candidates = [Checksum("a" * 64), Checksum("b" * 64)]
    if reverse:
        candidates.reverse()
    monkeypatch.setattr(database_remote, "has_read_database", lambda: False)
    monkeypatch.setattr(cache_mod.get_transformation_cache(), "get_reverse_transformations", lambda result: candidates)
    async def recompute(tf, **kwargs):
        if tf == "a" * 64:
            raise RuntimeError("failed candidate")
        return Checksum("e" * 64)
    monkeypatch.setattr(cache_mod, "recompute_from_transformation_checksum", recompute)
    with pytest.raises(CacheMissError) as caught:
        asyncio.run(Checksum("c" * 64).fingertip())
    assert caught.value.fingertip_category == FingertipCategory.IRREPRODUCIBLE_TRANSFORMATION


def test_nested_transformation_fingertip_keeps_category(monkeypatch):
    import uuid
    from seamless import FingertipCategory
    seamless.config.init()
    from seamless_transformer.transformation_cache import get_transformation_cache
    @delayed
    def inner_func(nonce) -> bytes:
        import os
        return os.urandom(32)
    inner_func.local = True
    inner_func.scratch = True
    inner = inner_func(uuid.uuid4().hex)
    inner_checksum = inner.construct()
    cache = get_transformation_cache()
    input_result = asyncio.run(cache.run(
        inner_checksum.resolve("plain"), tf_checksum=inner_checksum,
        tf_dunder=inner._tf_dunder, scratch=True, require_value=True,
    ))
    @delayed
    def outer(value, nonce) -> bytes:
        return value.content + nonce.encode()
    outer.scratch = True
    outer.local = True
    outer.celltypes.value = "bytes"
    outer.meta = {"allow_input_fingertip": True}
    tf = outer(input_result, uuid.uuid4().hex)
    result = tf.compute()
    assert isinstance(result, Checksum), tf.exception
    get_buffer_cache().purge_scratch(input_result)
    get_buffer_cache().purge_scratch(result)
    _drop_expression_buffer(input_result)
    _drop_expression_buffer(result)
    # Simulate both buffers being absent from every endpoint. Construction
    # may have published an input representation to the persistent hashserver.
    original_resolve, original_resolution = Checksum.resolve, Checksum.resolution
    missing = {input_result, result}
    def resolve(checksum, *args, **kwargs):
        if checksum in missing:
            raise CacheMissError(checksum)
        return original_resolve(checksum, *args, **kwargs)
    async def resolution(checksum, *args, **kwargs):
        if checksum in missing:
            raise CacheMissError(checksum)
        return await original_resolution(checksum, *args, **kwargs)
    monkeypatch.setattr(Checksum, "resolve", resolve)
    monkeypatch.setattr(Checksum, "resolution", resolution)
    with pytest.raises(CacheMissError) as caught:
        asyncio.run(result.fingertip())
    assert caught.value.fingertip_category == FingertipCategory.IRREPRODUCIBLE_TRANSFORMATION


@pytest.mark.parametrize("exception", [asyncio.CancelledError, ExecutionCanceledError, ClientConnectionError, ClientPayloadError, TimeoutError])
def test_fingertip_candidate_control_and_infrastructure_errors_propagate(monkeypatch, exception):
    import seamless_transformer.transformation_cache as cache_mod
    from seamless_remote import database_remote
    monkeypatch.setattr(database_remote, "has_read_database", lambda: False)
    monkeypatch.setattr(cache_mod.get_transformation_cache(), "get_reverse_transformations", lambda result: [Checksum("a" * 64)])
    async def canceled(*args, **kwargs):
        raise exception()
    monkeypatch.setattr(cache_mod, "recompute_from_transformation_checksum", canceled)
    with pytest.raises(exception):
        asyncio.run(Checksum("f" * 64).fingertip())


def test_fingertip_database_infrastructure_error_propagates(monkeypatch):
    from aiohttp import ClientConnectionError
    from seamless_remote import database_remote
    monkeypatch.setattr(database_remote, "has_read_database", lambda: True)
    async def unavailable(*args):
        raise ClientConnectionError("database unavailable")
    monkeypatch.setattr(database_remote, "get_rev_transformations", unavailable)
    with pytest.raises(ClientConnectionError):
        asyncio.run(Checksum("f" * 64).fingertip())
