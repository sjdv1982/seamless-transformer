from seamless import Buffer, Expression
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.transformer import delayed


def add_one(value):
    return value + 1


def test_expression_dependency_is_tempref_only_and_transformation_adopts_result():
    source = Buffer({"value": 4}, "plain")
    source_checksum = source.get_checksum()
    expression = Expression(source_checksum, "value", "plain", "int")
    transformation = delayed(add_one)(expression)
    assert transformation.compute() == Buffer(5, "int").get_checksum()
    expression_result = expression._result_checksum_internal()
    assert expression_result is None
    # Expression dependency evaluation is internal; the downstream
    # Transformation owns its concrete input role.
    assert any(role == "input:value" for _cs, role in transformation._refheld_checksums())
    transformation._release_refholds()

