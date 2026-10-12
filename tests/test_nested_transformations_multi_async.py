import asyncio
import os
import time

import seamless
from seamless.transformer import delayed
from tqdm import tqdm


def test_nested_transformations_multi():
    """Stress nested + nested-nested execution with many small jobs."""

    main_pid = os.getpid()
    job_count = 10

    try:

        @delayed
        def outer(label: str):
            from seamless.transformer import delayed, direct

            @direct
            def middle(label: str):
                from seamless.transformer import delayed, direct

                def leaf(label: str):
                    import time
                    import os

                    time.sleep(0.5)
                    return label, os.getpid()

                leaf = delayed(leaf)
                leaf.meta = {"process_lock": True}
                left_handle = leaf(f"{label}-a")
                assert left_handle.meta["process_lock"] is True
                left = left_handle.start()

                right = leaf(f"{label}-b").start()
                return left.run(), right.run()

            # The legacy lock enclosed only leaf bodies; the direct nesting
            # body must stay concurrent as well.
            middle.process_lock = False
            assert middle.meta["process_lock"] is False

            first = middle(f"{label}-1")
            second = middle(f"{label}-2")
            return first, second

        # The legacy lock enclosed only the leaf body. Keep outer setup and
        # nested-submission work overlapping while the leaf bodies serialize.
        outer.process_lock = False
        assert outer.meta["process_lock"] is False

        start = time.perf_counter()
        # tasks = [outer(f"job-{idx}").start() for idx in range(job_count)]    # top-level start

        async def get_results():
            tasks = [
                outer(f"job-{idx}").start() for idx in range(job_count)
            ]  # coro-level start
            with tqdm(total=len(tasks)) as progress:
                tasks2 = [t.task() for t in tasks]
                while 1:
                    done, pending = await asyncio.wait(tasks2, timeout=0.5)
                    progress.update(len(done) - progress.n)
                    if not len(pending):
                        break
            results = await asyncio.gather(*tasks2)
            return results

        results = results = asyncio.run(get_results())
        duration = time.perf_counter() - start

        expected_labels = {
            f"job-{idx}-{suffix}"
            for idx in range(job_count)
            for suffix in ("1-a", "1-b", "2-a", "2-b")
        }

        seen_labels = set()
        for outer_res in results:
            for mid in outer_res:
                for leaf_res in mid:
                    label, pid = leaf_res
                    seen_labels.add(label)
                    assert pid == main_pid

        assert expected_labels == seen_labels
        assert duration < 20.0 + 2
    finally:
        seamless.close()
