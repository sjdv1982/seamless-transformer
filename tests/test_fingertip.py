import asyncio

import pytest

from seamless import CacheMissError, Checksum, Expression
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

    result_checksum.tempref(scratch=True)
    get_buffer_cache().purge_scratch(result_checksum)

    with pytest.raises(CacheMissError):
        result_checksum.resolve()

    value = asyncio.run(result_checksum.fingertip("mixed"))
    assert value == pytest.approx(12.84)

    buf = result_checksum.resolve()
    assert buf.get_value("mixed") == pytest.approx(12.84)

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
    expression = Expression(tf_result, "a", "plain", "str")
    expression_result = expression.compute()

    tf_result.tempref(scratch=True)
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
