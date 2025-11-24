import time
import asyncio

from seamless_transformer import transformer

DELAY = 0.5


def test_transformation_async():
    @transformer(return_transformation=True)
    def func(a, b, delay):
        import time

        time.sleep(delay)
        return 10 * a + 2 * b

    async def main():
        tasks = [func(n, -1, DELAY).task() for n in range(4)]
        results = await asyncio.gather(*tasks)
        return results

    start = time.perf_counter()
    results = asyncio.run(main())
    duration1 = time.perf_counter() - start

    for tasknr, result in enumerate(results):
        assert result == 10 * tasknr - 2
    duration2 = time.perf_counter() - start
    assert duration2 - duration1 < 1
