import pytest
import time
import asyncio

from seamless.transformer import direct, delayed
import seamless.config

DELAY = 0.5
N = 40  # make this no bigger than the WORKERS, or increase PARALLEL_FACTOR
# COMMENT: beyond N=80 or so, the system starts to run into slowdowns...
#   => increase PARALLEL_FACTOR beyond N/WORKERS,e.g. to 100 for N=400
#   => need seamless-multi pools
PARALLEL_FACTOR = 5


@pytest.fixture(scope="session", autouse=True)
def _close_seamless_session():
    """Ensure Seamless shuts down once after the full test session."""
    import seamless

    yield
    seamless.close()


def test_spawn_config():
    seamless.config.set_stage("spawn-config")
    seamless.config.init()

    @delayed
    def func(a, b, delay):
        import time

        time.sleep(delay)
        return 10000 * a + 3 * b

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
        assert result == 10000 * tasknr - 3
    duration2 = time.perf_counter() - start
    assert duration2 - duration1 < 1
