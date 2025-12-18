import time
import asyncio

import seamless.config

seamless.config.init()

from seamless.transformer import direct, delayed

DELAY = 0.5

NSTART = 1  # increment this to force cache misses
N = 1000  # not beyond 10 000


# TODO: investigate why every transformation base task takes hundreds of millisecs,
#   even with no global_lock and no delay


def test_transformation_async():
    @delayed
    def func(a, b, delay):
        import time
        from seamless.transformer import global_lock

        with global_lock:
            time.sleep(delay)
        return 2 * b + 10 * a

    async def main():
        tasks = [func(NSTART * 10000 + n, -1, DELAY).task() for n in range(N)]
        results = await asyncio.gather(*tasks)
        return results

    start = time.perf_counter()
    results = asyncio.run(main())
    duration1 = time.perf_counter() - start
    print(f"Duration for {N} calls", duration1)

    for tasknr, result in enumerate(results):
        assert result == 10 * (tasknr + NSTART * 10000) - 2
