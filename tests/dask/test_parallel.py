import asyncio
import os
import time

import seamless.config

seamless.config.init()

from seamless.transformer import TransformationList, delayed, parallel, parallel_async

DELAY = float(os.environ.get("SEAMLESS_TEST_ASYNC_DELAY", "0.2"))


def test_parallel_dask():
    nonce = time.time_ns()

    @delayed
    def func(a, b, delay, nonce):
        import time

        time.sleep(delay)
        return nonce + 10 * a + b

    n = 20
    tfs = [func(i, -1, DELAY / 4, nonce) for i in range(n)]
    for i, tf in enumerate(parallel(tfs)):
        assert tf.result_checksum
        assert tf.value == nonce + 10 * i - 1


def test_parallel_async_dask():
    nonce = time.time_ns()

    @delayed
    def func(a, b, delay, nonce):
        import time

        time.sleep(delay)
        return nonce + 100 * a + b

    async def main():
        tfs = [func(i, -3, DELAY / 4, nonce) for i in range(16)]
        results = []
        async for tf in parallel_async(tfs):
            results.append(tf)
        return results

    tfs = asyncio.run(main())

    for i, tf in enumerate(tfs):
        assert tf.result_checksum
        assert tf.value == nonce + 100 * i - 3


def test_parallel_transformation_list_dask():
    nonce = time.time_ns()

    @delayed
    def func(a, nonce):
        return nonce + a + 5

    tflist = TransformationList(
        [func(i, nonce) for i in range(18)],
        show_progress=True,
    )
    start = time.perf_counter()
    for i, tf in enumerate(parallel(tflist)):
        assert tf.value == nonce + i + 5
    duration = time.perf_counter() - start
    print(f"Duration for TransformationList dask parallel(): {duration}")

    assert tflist._finished == 18
    assert tflist._errors == 0
    assert tflist._cancelled == 0
    assert tflist._pbar is not None
    assert tflist._pbar.n == 18
    assert tflist._pbar.total == 18
    assert tflist[7].value == nonce + 12
