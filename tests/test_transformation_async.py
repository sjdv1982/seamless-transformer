import time
import asyncio

from seamless.transformer import direct, delayed

DELAY = 0.5
N = 10


def test_transformation_async():
    @delayed
    def func(a, b, delay):
        import time
        from seamless.transformer import global_lock

        with global_lock:
            time.sleep(delay)
        return 10 * a + 2 * b + 0

    async def main():
        tasks = [func(n, -1, DELAY).task() for n in range(N)]
        results = await asyncio.gather(*tasks)
        return results

    start = time.perf_counter()
    results = asyncio.run(main())
    duration1 = time.perf_counter() - start
    print(f"Duration for {N} calls", duration1)
    assert DELAY * N < duration1 < DELAY * N * 2.5

    for tasknr, result in enumerate(results):
        assert result == 10 * tasknr - 2
    duration2 = time.perf_counter() - start
    assert duration2 - duration1 < 1
