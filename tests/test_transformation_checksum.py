import seamless.config
import asyncio
import pytest

seamless.config.set_stage("persistent-test")
seamless.config.init()

from seamless.transformer import delayed
from seamless_transformer.pretransformation import PreTransformation
from seamless_transformer.transformation_class import transformation_from_pretransformation
from seamless_transformer.transformation_utils import sufficiently_connected


def test_transformation_checksum():
    """Print out a transformation checksum, to be used with seamless-run-transformation"""

    @delayed
    def func(a, b, random_value) -> int:
        return a + b

    import numpy as np

    tf = func(12, 13, np.random.random())
    print(tf.construct())
    print(tf.exception)


def _checksum_for(pretransformation_dict, *, optional_pins=()):
    pre = PreTransformation(
        pretransformation_dict,
        optional_pins=optional_pins,
    )
    tf = transformation_from_pretransformation(
        pre,
        upstream_dependencies={},
        meta={},
        scratch=False,
        tf_dunder={},
    )
    checksum = tf.construct()
    assert tf.exception is None, tf.exception
    assert checksum is not None
    return checksum.hex()


def _base_dict():
    return {
        "__language__": "python",
        "__output__": ("result", "plain", None),
        "code": ("python", "transformer", "result = a\n"),
        "a": ("plain", None, 10),
    }


def test_optional_pin_absent_and_json_null_have_same_identity():
    absent = _checksum_for(_base_dict(), optional_pins={"x"})

    null_connected = _base_dict()
    null_connected["x"] = ("plain", None, None)
    null_checksum = _checksum_for(null_connected, optional_pins={"x"})

    assert null_checksum == absent


def test_non_null_optional_pin_participates_in_identity():
    absent = _checksum_for(_base_dict(), optional_pins={"x"})

    non_null = _base_dict()
    non_null["x"] = ("plain", None, 123)
    non_null_checksum = _checksum_for(non_null, optional_pins={"x"})

    assert non_null_checksum != absent


def test_required_json_null_remains_present_in_identity():
    absent_optional = _checksum_for(_base_dict(), optional_pins={"x"})

    required_null = _base_dict()
    required_null["x"] = ("plain", None, None)
    required_null_checksum = _checksum_for(required_null)

    assert required_null_checksum != absent_optional


def test_optional_json_null_rejects_non_null_encodable_celltype():
    invalid = _base_dict()
    invalid["x"] = ("binary", None, None)
    pre = PreTransformation(invalid, optional_pins={"x"})
    with pytest.raises(
        TypeError,
        match="Optional pin 'x' with celltype 'binary' cannot use JSON null as absence",
    ):
        transformation_from_pretransformation(
            pre,
            upstream_dependencies={},
            meta={},
            scratch=False,
            tf_dunder={},
        )


def test_sufficiently_connected_ignores_unwired_optional_pins():
    assert sufficiently_connected({"a", "x"}, {"x"}, {"a"})
    assert sufficiently_connected({"a", "x"}, {"x"}, {"a", "x"})
    assert not sufficiently_connected({"a", "x"}, {"x"}, set())


def test_connected_optional_dependency_failure_is_not_absence():
    @delayed
    def boom():
        raise RuntimeError("upstream failed")

    @delayed
    def consume(a, x=None):
        return a

    consume.optional_pins.add("x")
    tf = consume(1, boom())

    assert tf.construct() is None
    assert "Dependency 'x' has an exception" in tf.exception
    assert "upstream failed" in tf.exception


def test_connected_optional_dependency_failure_is_not_absence_async():
    @delayed
    def boom():
        raise RuntimeError("async upstream failed")

    @delayed
    def consume(a, x=None):
        return a

    consume.optional_pins.add("x")
    tf = consume(1, boom())

    async def run_construction():
        return await tf.construction()

    assert asyncio.run(run_construction()) is None
    assert "Dependency 'x' has an exception" in tf.exception
    assert "async upstream failed" in tf.exception
