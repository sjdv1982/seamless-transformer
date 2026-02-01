import asyncio

import pytest

from seamless import CacheMissError, Checksum
import seamless.config
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.transformer import delayed


def _disable_remote() -> None:
    try:
        from seamless_config.select import (
            select_execution,
            select_persistent,
            select_remote,
        )
    except Exception:
        select_execution = select_persistent = select_remote = None
    if select_execution is not None:
        select_execution("process")
    if select_persistent is not None:
        select_persistent(False)
    if select_remote is not None:
        select_remote("daskserver")
    try:
        from seamless_remote import (
            buffer_remote,
            database_remote,
            daskserver_remote,
            jobserver_remote,
        )
    except Exception:
        return
    buffer_remote.DISABLED = True
    database_remote.DISABLED = True
    daskserver_remote.DISABLED = True
    jobserver_remote.DISABLED = True


def test_fingertip_recompute_scratch():
    _disable_remote()
    seamless.config.init()

    @delayed
    def func(a, b) -> float:
        return 2.13 * a + 2.86 * b

    func.scratch = True
    func.local = True

    tf = func(2, 3)
    result_checksum = tf.compute()
    assert isinstance(result_checksum, Checksum), tf.exception

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
    _disable_remote()
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
