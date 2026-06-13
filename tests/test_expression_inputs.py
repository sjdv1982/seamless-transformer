import pytest

from seamless import Buffer, Expression
from seamless.transformer import delayed
from seamless_transformer.transformation_class import TransformationError


def _checksum(value, celltype="plain"):
    return Buffer(value, celltype).get_checksum()


def test_transformation_consumes_expression_result():
    @delayed
    def add_one(value: int) -> int:
        return value + 1

    add_one.local = True
    expr = Expression(_checksum({"value": 41}), "value", "plain", "int")

    assert add_one(expr).run() == 42


def test_transformation_consumes_expression_over_transformation_result():
    @delayed
    def make_record(value: int):
        return {"value": value}

    @delayed
    def add_one(value: int) -> int:
        return value + 1

    make_record.local = True
    add_one.local = True
    expr = Expression(make_record(41), "value", "plain", "int")

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
    first = Expression(make_record(21), "nested", "plain", "plain")
    second = Expression(first, "value", "plain", "int")

    assert double(second).run() == 42


def test_expression_dependency_failure_blocks_transformation():
    @delayed
    def add_one(value: int) -> int:
        return value + 1

    add_one.local = True
    expr = Expression(_checksum({"value": 41}), "missing", "plain", "int")
    transformation = add_one(expr)

    with pytest.raises(TransformationError) as exc_info:
        transformation.run()

    message = str(exc_info.value)
    assert "Dependency 'value' has an exception" in message
    assert "missing" in message
