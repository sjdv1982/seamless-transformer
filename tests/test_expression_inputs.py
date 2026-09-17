import pytest

from seamless import Buffer, Expression
from seamless.transformer import delayed
from seamless_transformer.pretransformation import (
    PreTransformation,
    PreparedPreTransformation,
)
from seamless_transformer.transformation_class import TransformationError


def _checksum(value, celltype="plain"):
    return Buffer(value, celltype).get_checksum()


def test_transformation_consumes_expression_result():
    @delayed
    def add_one(value: int) -> int:
        return value + 1

    add_one.local = True
    expr = Expression(_checksum({"value": 41}), "value", input_celltype="plain", celltype="int")

    assert add_one(expr).run() == 42


def test_transformation_expression_dependency_uses_auto(monkeypatch):
    executions = []
    original_evaluate = Expression._evaluate_internal

    def record_execution(self, *, execution):
        executions.append(execution)
        return original_evaluate(self, execution=execution)

    monkeypatch.setattr(Expression, "_evaluate_internal", record_execution)

    @delayed
    def add_one(value: int) -> int:
        return value + 1

    add_one.local = True
    source = Buffer({"value": 41}, "plain")
    source.tempref()
    expr = Expression(
        source.get_checksum(),
        "value",
        input_celltype="plain",
        celltype="int",
    )

    assert add_one(expr).run() == 42
    assert executions
    assert set(executions) == {"auto"}


@pytest.mark.parametrize(
    "pretransformation_cls",
    [PreTransformation, PreparedPreTransformation],
)
def test_pretransformation_expression_paths_request_auto(
    monkeypatch, pretransformation_cls
):
    expected = _checksum(43, "int")
    executions = []

    def record_execution(self, *, execution):
        executions.append(execution)
        return expected

    monkeypatch.setattr(Expression, "_evaluate_internal", record_execution)
    expr = Expression(
        _checksum({"value": 43}),
        "value",
        input_celltype="plain",
        celltype="int",
    )
    pretransformation = pretransformation_cls({"__language__": "python"})

    assert pretransformation._prepare_pin_value("value", expr, "int") == expected
    assert executions == ["auto"]


def test_transformation_consumes_expression_over_transformation_result():
    @delayed
    def make_record(value: int):
        return {"value": value}

    @delayed
    def add_one(value: int) -> int:
        return value + 1

    make_record.local = True
    add_one.local = True
    expr = Expression(make_record(41), "value", celltype="int")

    assert add_one(expr).run() == 42


def test_expression_result_feeds_another_transformation():
    @delayed
    def make_record(value: int):
        return {"nested": {"value": value}}

    @delayed
    def double(value: int) -> int:
        return value * 2

    make_record.local = True
    double.local = True
    first = Expression(make_record(21), "nested", celltype="plain")
    second = Expression(first, "value", input_celltype="plain", celltype="int")

    assert double(second).run() == 42


def test_expression_dependency_failure_blocks_transformation():
    @delayed
    def add_one(value: int) -> int:
        return value + 1

    add_one.local = True
    expr = Expression(_checksum({"value": 41}), "missing", input_celltype="plain", celltype="int")
    transformation = add_one(expr)

    with pytest.raises(TransformationError) as exc_info:
        transformation.run()

    message = str(exc_info.value)
    assert "Dependency 'value' has an exception" in message
    assert "missing" in message


def test_async_transformation_expression_dependency_uses_auto(monkeypatch):
    import asyncio
    from seamless_transformer.transformation_class import _dependency_computation

    executions = []
    original = Expression._evaluate_internal_async

    async def record(self, *, execution):
        executions.append(execution)
        return await original(self, execution=execution)

    monkeypatch.setattr(Expression, "_evaluate_internal_async", record)
    source = Buffer({"value": 43}, "plain")
    source_ref = source.tempref()
    expr = Expression(source.get_checksum(), "value", input_celltype="plain", celltype="int")
    try:
        result = asyncio.run(_dependency_computation(expr, require_value=False))
        assert result == Buffer(43, "int").get_checksum()
        assert executions == ["auto"]
    finally:
        source_ref.clear()
