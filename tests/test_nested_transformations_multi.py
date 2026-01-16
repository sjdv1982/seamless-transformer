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
                    from seamless.transformer import global_lock

                    with global_lock:
                        time.sleep(0.5)
                    return label, os.getpid()

                leaf = delayed(leaf)

                left = leaf(f"{label}-a").start()

                right = leaf(f"{label}-b").start()
                return left.run(), right.run()

            first = middle(f"{label}-1")
            second = middle(f"{label}-2")
            return first, second

        start = time.perf_counter()
        tasks = [outer(f"job-{idx}").start() for idx in range(job_count)]
        results = [tf.run() for tf in tqdm(tasks)]
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
        assert duration < 2.0 * job_count + 2
    finally:
        seamless.close()
