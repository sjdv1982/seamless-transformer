"""Pin ownership from contracts/pins.md, *Scratch at the pin*."""

import uuid

import pytest

from seamless import Buffer, Checksum, Expression
from seamless.caching import buffer_writer
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.transformer import delayed
from seamless_transformer.transformation_class import transformation_from_dict


def identity(value):
    return value


def add_suffix(value):
    return value + "-result"


@pytest.fixture
def writes(monkeypatch):
    result = []
    monkeypatch.setattr(buffer_writer, "register", lambda buf: result.append(buf.get_checksum()))
    return result


def test_literal_stays_published_with_input_fingertip(writes):
    """pins.md, *Scratch at the pin*: a literal pin always publishes."""
    builder = delayed(identity)
    builder.scratch = True
    builder.allow_input_fingertip = True
    value = "literal-" + uuid.uuid4().hex
    builder.pins.value = value
    value_buffer = Buffer(value, "mixed")
    checksum = value_buffer.get_checksum()
    tf = builder(value)
    try:
        assert checksum in writes
        assert (checksum, "input:value") in tf._refheld_checksums()
        assert tf._pin_scratch("value") is False
    finally:
        tf._release_refholds()
        builder._release_refholds()


@pytest.mark.parametrize("allow_fingertip", [False, True])
def test_expression_pin_requests_its_own_scratch(monkeypatch, allow_fingertip):
    """pins.md, *Scratch at the pin*: an Expression carries its pin's scratch."""
    calls = []
    original = Expression._evaluate_internal

    def record(self, *, execution, scratch, materialize):
        calls.append((scratch, materialize))
        return original(self, execution=execution, scratch=scratch,
                        materialize=materialize)

    monkeypatch.setattr(Expression, "_evaluate_internal", record)
    source = Buffer({"value": "v-" + uuid.uuid4().hex}, "plain")
    source.tempref()
    expr = Expression(source.get_checksum(), "value", input_celltype="plain",
                      celltype="mixed")
    builder = delayed(identity)
    builder.local = True
    builder.allow_input_fingertip = allow_fingertip
    tf = builder(expr)
    try:
        assert tf.run() is not None
        assert (allow_fingertip, not allow_fingertip) in calls
    finally:
        tf._release_refholds()
        builder._release_refholds()


@pytest.mark.parametrize("allow_fingertip", [False, True])
def test_expression_pin_chain_bans_intermediate(monkeypatch, allow_fingertip):
    from seamless.checksum import expression as expression_mod

    calls = []
    original = expression_mod.evaluate_expression_placed

    async def record(input_checksum, path, input_celltype, celltype, **kwargs):
        calls.append((path, input_celltype, celltype,
                      kwargs["scratch"], kwargs["materialize"]))
        return await original(input_checksum, path, input_celltype, celltype, **kwargs)

    monkeypatch.setattr(expression_mod, "evaluate_expression_placed", record)
    source = Buffer('{"x": "value"}', "text")
    source.tempref()
    conversion = Expression(
        source.get_checksum(), input_celltype="text", celltype="plain",
    )
    expr = Expression(conversion, path="x", input_celltype="plain", celltype="plain")
    builder = delayed(identity)
    builder.local = True
    builder.celltypes.value = "plain"
    builder.allow_input_fingertip = allow_fingertip
    tf = builder(expr)
    try:
        assert tf.run() == "value"
        assert ("", "text", "plain", True, False) in calls
        assert ("x", "plain", "plain", allow_fingertip,
                not allow_fingertip) in calls
    finally:
        tf._release_refholds()
        builder._release_refholds()


@pytest.mark.parametrize("path,expected_override", [("", False), ("x", None)])
def test_cell_overrules_scratch_transformer_only_for_own_buffer(
    monkeypatch, path, expected_override,
):
    from seamless import Cell
    from seamless_transformer.transformation_class import Transformation

    builder = delayed(identity)
    builder.local = True
    builder.scratch = True
    producer = builder({"x": "value"})
    calls = []
    original = Transformation._compute_dependency_async

    async def record(self, *, require_value=False, scratch_override=None):
        calls.append(scratch_override)
        return await original(
            self, require_value=require_value, scratch_override=scratch_override,
        )

    monkeypatch.setattr(Transformation, "_compute_dependency_async", record)
    source = (Expression(producer, path=path, input_celltype="mixed",
                         celltype="mixed") if path else producer)
    cell = Cell("mixed", source=source)
    try:
        assert cell.compute() is not None
        assert expected_override in calls
    finally:
        cell._release_refholds()
        producer._release_refholds()
        builder._release_refholds()


@pytest.mark.parametrize("allow_fingertip", [False, True])
def test_local_scratch_producer_is_overruled_by_non_scratch_pin(writes, allow_fingertip):
    """pins.md, *Scratch at the pin*: a non-scratch pin publishes a local producer."""
    producer_builder = delayed(add_suffix)
    producer_builder.local = True
    producer_builder.scratch = True
    producer = producer_builder("producer-" + uuid.uuid4().hex)
    consumer_builder = delayed(identity)
    consumer_builder.local = True
    consumer_builder.scratch = True
    consumer_builder.allow_input_fingertip = allow_fingertip
    consumer = consumer_builder(producer)
    try:
        assert consumer.run() is not None
        checksum = producer.result_checksum
        assert (checksum in writes) is (not allow_fingertip)
    finally:
        consumer._release_refholds()
        producer._release_refholds()
        consumer_builder._release_refholds()
        producer_builder._release_refholds()


@pytest.mark.parametrize("allow_fingertip", [False, True])
def test_prepared_dict_reads_input_fingertip_meta(writes, allow_fingertip):
    """pins.md, *Scratch at the pin*: replayed pins use the dict's opt-in."""
    value = "prepared-" + uuid.uuid4().hex
    value_buffer = Buffer(value, "mixed")
    checksum = value_buffer.get_checksum()
    checksum.tempref()
    checksum.mark_scratch()
    code = Buffer("def transform(value):\n    return value\n", "python")
    code.tempref()
    transformation_dict = {
        "__language__": "python",
        "__output__": ("result", "mixed", None),
        "__meta__": {"allow_input_fingertip": allow_fingertip},
        "code": ("python", "transformer", code.get_checksum().hex()),
        "value": ("mixed", None, checksum.hex()),
    }
    tf = transformation_from_dict(transformation_dict)
    try:
        assert tf.allow_input_fingertip is allow_fingertip
        assert tf._pin_scratch("value") is allow_fingertip
        assert (checksum in writes) is (not allow_fingertip)
    finally:
        tf._release_refholds()


def test_two_consumers_only_non_scratch_pin_publishes_shared_result(writes):
    """pins.md, *Scratch at the pin*: each consumer has its own pin policy."""
    producer_builder = delayed(add_suffix)
    producer_builder.local = True
    producer_builder.scratch = True
    producer = producer_builder("shared-" + uuid.uuid4().hex)
    scratch_builder = delayed(add_suffix)
    scratch_builder.local = True
    scratch_builder.scratch = True
    scratch_builder.allow_input_fingertip = True
    scratch_consumer = scratch_builder(producer)
    publishing_builder = delayed(add_suffix)
    publishing_builder.local = True
    publishing_builder.scratch = True
    publishing_consumer = publishing_builder(producer)
    try:
        assert scratch_consumer.run() is not None
        result = producer.result_checksum
        assert result not in writes
        assert publishing_consumer.run() is not None
        assert writes.count(result) == 1
        assert scratch_consumer._pin_scratch("value") is True
        assert publishing_consumer._pin_scratch("value") is False
    finally:
        for holder in (scratch_consumer, publishing_consumer, producer,
                       scratch_builder, publishing_builder, producer_builder):
            holder._release_refholds()


def test_recorded_scratch_dependency_is_rerun_for_value(monkeypatch):
    """pins.md, *Scratch at the pin*: a checksum without bytes cannot answer a value request."""
    from seamless_transformer import transformation_cache

    producer_builder = delayed(add_suffix)
    producer_builder.local = True
    producer_builder.scratch = True
    value = "remote-" + uuid.uuid4().hex
    source = Buffer(value, "mixed")
    source_checksum = source.get_checksum().hex()
    result_value = value + "-result"
    result_buffer = Buffer(result_value, "mixed")
    result = result_buffer.get_checksum()
    cache = get_buffer_cache()
    with cache.lock:
        cache.weak_cache.pop(result, None)
        cache.strong_cache.pop(result, None)
    calls = []
    original_sync = transformation_cache.run_sync
    original_async = transformation_cache.run

    def fake_run(transformation_dict, *, tf_checksum, scratch, **kwargs):
        if transformation_dict["value"][2] == source_checksum:
            calls.append(scratch)
            if not scratch:
                cache.register(result, result_buffer)
            return result
        return original_sync(transformation_dict, tf_checksum=tf_checksum,
                             scratch=scratch, **kwargs)

    async def fake_run_async(transformation_dict, *, tf_checksum, scratch, **kwargs):
        if transformation_dict["value"][2] == source_checksum:
            calls.append(scratch)
            if not scratch:
                cache.register(result, result_buffer)
            return result
        return await original_async(transformation_dict, tf_checksum=tf_checksum,
                                    scratch=scratch, **kwargs)

    monkeypatch.setattr(transformation_cache, "run_sync", fake_run)
    monkeypatch.setattr(transformation_cache, "run", fake_run_async)
    producer = producer_builder(value)
    consumer_builder = delayed(identity)
    consumer_builder.local = True
    consumer_builder.scratch = True
    consumer = consumer_builder(producer)
    try:
        assert producer._compute_dependency() == result
        assert consumer.run() == result_value
        assert calls == [True, False]
    finally:
        for holder in (consumer, producer, consumer_builder, producer_builder):
            holder._release_refholds()


@pytest.mark.parametrize("allow_fingertip", [False, True])
def test_standalone_pin_fingertip_uses_pin_claim(writes, monkeypatch, allow_fingertip):
    """pins.md, *Scratch at the pin*: fingertipping persists for a non-scratch pin."""
    value = "finger-" + uuid.uuid4().hex
    source = Buffer({"value": value}, "plain")
    source.tempref()
    expr = Expression(source.get_checksum(), "value", input_celltype="plain",
                      celltype="plain")
    builder = delayed(identity)
    builder.allow_input_fingertip = allow_fingertip
    builder.celltypes.value = "plain"
    builder.pins.value = expr
    pin = builder.pins.value
    try:
        checksum = pin.checksum
        assert checksum is not None, pin.exception
        recovered = Buffer(value, "plain")
        assert recovered.get_checksum() == checksum
        cache = get_buffer_cache()
        with cache.lock:
            cache.weak_cache.pop(checksum, None)
            cache.strong_cache[checksum].buffer = None
        writes.clear()
        original = Checksum.fingertip_sync

        def fingertip(self, celltype=None):
            if self == checksum:
                cache.register(checksum, recovered)
                return recovered
            return original(self, celltype)

        monkeypatch.setattr(Checksum, "fingertip_sync", fingertip)
        assert pin.fingertip() is recovered
        assert (checksum in writes) is (not allow_fingertip)
    finally:
        builder._release_refholds()
