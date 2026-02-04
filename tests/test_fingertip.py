import asyncio

import pytest

from seamless import CacheMissError, Checksum
import seamless.config
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.transformer import delayed

from seamless.config import set_stage

set_stage("fingertip")


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
