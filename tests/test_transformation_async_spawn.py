import pytest
import time
import asyncio

from seamless.transformer import direct, delayed
from seamless.transformer import spawn, has_spawned

DELAY = 0.5
WORKERS = 8  # make this no bigger than #cores on your system
N = 640  # make this no bigger than the WORKERS, or increase PARALLEL_FACTOR to N/WORKERS
PARALLEL_FACTOR = 80


@pytest.fixture(scope="module")
def spawned_workers():
    if not has_spawned():
        spawn(WORKERS)
    yield True


@pytest.fixture(scope="session", autouse=True)
def _close_seamless_session():
    """Ensure Seamless shuts down once after the full test session."""
    import seamless

    yield
    seamless.close()


def test_transformation_async(spawned_workers):
    @delayed
    def func(a, b, delay):
        from seamless_transformer import global_lock
        import time

        with global_lock:
            time.sleep(delay)
        return 10000 * a + 2 * b

    async def main():
        tasks = [func(n, -1, DELAY).task() for n in range(N)]
        report_progress = False
        if N >= 100:
            try:
                import tqdm

                report_progress = True
            except ImportError:
                pass
        if report_progress:
            with tqdm.tqdm(total=N) as progress:
                while 1:
                    done, pending = await asyncio.wait(tasks, timeout=0.5)
                    progress.update(len(done) - progress.n)
                    if not len(pending):
                        break
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
