import asyncio
import gc
import math
import time
import weakref

import pytest
import seamless.config

from seamless.transformer import (
    TransformationList,
    delayed,
    has_spawned,
    parallel,
    parallel_async,
    spawn,
)

DELAY = 0.2
WORKERS = 8


@pytest.fixture(scope="module")
def spawned_workers():
    if not has_spawned():
        spawn(WORKERS)
    yield True


def test_parallel_basic(spawned_workers):
    seamless.config.set_nparallel(4)
    nonce = time.time_ns()

    @delayed
    def func(a, b, delay, nonce):
        import time

        time.sleep(delay)
        return nonce + 10000 * a + 2 * b

    n = 24
    tfs = [func(i, -1, DELAY / 4, nonce) for i in range(n)]
    for i, tf in enumerate(parallel(tfs)):
        assert tf.result_checksum
        assert tf.value == nonce + 10000 * i - 2


def test_parallel_nparallel_cap(spawned_workers):
    seamless.config.set_nparallel(2)
    nonce = time.time_ns()

    @delayed
    def func(a, delay, nonce):
        import time

        time.sleep(delay)
        return nonce + a

    n = 6
    tfs = [func(i, DELAY, nonce) for i in range(n)]
    start = time.perf_counter()
    results = [tf.value for tf in parallel(tfs)]
    duration = time.perf_counter() - start
    print(f"Duration for {n} parallel() calls at nparallel=2: {duration}")

    assert duration >= DELAY * math.ceil(n / 2) * 0.7
    assert results == [nonce + i for i in range(n)]


def test_parallel_async_basic(spawned_workers):
    seamless.config.set_nparallel(4)
    nonce = time.time_ns()

    @delayed
    def func(a, b, delay, nonce):
        import time

        time.sleep(delay)
        return nonce + 10 * a + b

    async def main():
        tfs = [func(i, -1, DELAY / 4, nonce) for i in range(18)]
        results = []
        async for tf in parallel_async(tfs):
            results.append(tf)
        return results

    tfs = asyncio.run(main())

    for i, tf in enumerate(tfs):
        assert tf.result_checksum
        assert tf.value == nonce + 10 * i - 1


def test_parallel_with_transform_list(spawned_workers):
    seamless.config.set_nparallel(4)
    nonce = time.time_ns()

    @delayed
    def func(a, nonce):
        return nonce + a + 1

    tfs = [func(i, nonce) for i in range(20)]
    tflist = TransformationList(tfs, show_progress=False)
    for i, tf in enumerate(parallel(tflist)):
        assert tf.value == nonce + i + 1

    assert len(tflist) == 20
    assert len(tflist[:3]) == 3
    assert tflist[5].value == nonce + 6
    assert tflist._finished == 20
    assert tflist._errors == 0
    assert tflist._cancelled == 0


def test_parallel_streams_prefix(spawned_workers):
    seamless.config.set_nparallel(2)
    nonce = time.time_ns()

    @delayed
    def func(a, delay, nonce):
        import time

        time.sleep(delay)
        return nonce + a

    delays = [0.2, 0.2, 0.6, 0.6]
    stream = parallel([func(i, delays[i], nonce) for i in range(len(delays))])
    start = time.perf_counter()
    first = next(stream)
    first_duration = time.perf_counter() - start
    rest = [tf.value for tf in stream]

    assert first.value == nonce
    assert first_duration < 0.45
    assert rest == [nonce + 1, nonce + 2, nonce + 3]


def test_parallel_error_handling(spawned_workers, capsys):
    seamless.config.set_nparallel(4)

    @delayed
    def func(a):
        if a % 3 == 0:
            raise ValueError(f"boom-{a}")
        return a

    printed = TransformationList(
        [func(i) for i in range(8)],
        show_progress=False,
        on_error="print",
    )
    list(parallel(printed))
    captured = capsys.readouterr()

    assert printed._finished == 8
    assert printed._errors == 3
    assert "boom-0" in captured.out
    assert "boom-3" in captured.out
    assert "boom-6" in captured.out

    raising = TransformationList(
        [func(i) for i in range(8)],
        show_progress=False,
        on_error="raise",
    )
    with pytest.raises(RuntimeError, match=r"Transformation \d+ failed"):
        list(parallel(raising))
    assert raising._errors == 1


def test_parallel_store_details_releases_transformations(spawned_workers):
    seamless.config.set_nparallel(4)

    @delayed
    def func(a):
        return a + 100

    transformations = [func(i) for i in range(40)]
    refs = [weakref.ref(tf) for tf in transformations]
    tflist = TransformationList(
        transformations,
        show_progress=False,
        store_details=True,
    )
    del transformations

    list(parallel(tflist))
    gc.collect()

    assert tflist._finished == 40
    assert all(tf is None for tf in tflist._transformations)
    assert len(tflist._stored_checksums) == 40
    assert len(tflist._stored_exceptions) == 40
    assert all(exc is None for exc in tflist._stored_exceptions.values())
    assert all(checksum is not None for checksum in tflist._stored_checksums.values())
    assert all(ref() is None for ref in refs)
