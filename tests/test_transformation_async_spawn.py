import pytest
import time
import asyncio

from seamless_transformer import transformer
from seamless_transformer.worker import spawn, has_spawned, shutdown_workers

DELAY = 0.5
WORKERS = 8  # make this no bigger than #cores on your system
N = 400  # make this no bigger than the WORKERS, or increase PARALLEL_FACTOR
# COMMENT: beyond N=80 or so, the system starts to run into slowdowns...
#   => increase PARALLEL_FACTOR beyond N/WORKERS,e.g. to 100 for N=400
#   => need seamless-multi pools
PARALLEL_FACTOR = 100


@pytest.fixture(scope="session")
def spawned_workers():
    if not has_spawned:
        spawn(WORKERS)
    yield True
    shutdown_workers()


def test_transformation_async(spawned_workers):
    @transformer(return_transformation=True)
    def func(a, b, delay):
        import time

        time.sleep(delay)
        return 10000 * a + 2 * b

    async def main():
        tasks = [func(n, -1, DELAY).task() for n in range(N)]
        results = await asyncio.gather(*tasks)
        return results

    start = time.perf_counter()
    results = asyncio.run(main())
    duration1 = time.perf_counter() - start
    print(f"Duration for {N} calls", duration1)
    assert duration1 < DELAY * PARALLEL_FACTOR + 1  # parallel execution

    for tasknr, result in enumerate(results):
        assert result == 10000 * tasknr - 2
    duration2 = time.perf_counter() - start
    assert duration2 - duration1 < 1
