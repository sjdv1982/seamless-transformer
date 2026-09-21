"""Sync API called from a thread whose event loop is running (the Jupyter condition).

ipykernel runs cell code synchronously inside a callback of the main-thread
asyncio loop. ``asyncio.run(cell())`` with a synchronous body reproduces that
without a kernel.
"""

import asyncio
import os
import threading
import time

from seamless.transformer import delayed, direct


def in_running_loop(func):
    async def cell():
        return func()

    return asyncio.run(cell())


def test_sync_entry_points():
    @direct
    def add_direct(a, b):
        return a + b

    @delayed
    def add(a, b):
        return a + b

    def body():
        results = {"direct": add_direct(2, 3), "run": add(20, 30).run()}
        tf = add(200, 300)
        tf.compute()
        results["compute"] = tf.value
        results["dependency"] = add(add(1, 2), add(3, 4)).run()
        return results

    assert in_running_loop(body) == {
        "direct": 5,
        "run": 50,
        "compute": 500,
        "dependency": 10,
    }


def test_start_in_loop_then_sync_run():
    @delayed
    def add(a, b):
        return a + b

    def body():
        tf = add(7, 8).start()
        return tf.run()

    assert in_running_loop(body) == 15


def test_start_in_loop_then_async_api():
    @delayed
    def add(a, b):
        return a + b

    async def cell():
        tf1 = add(70, 80).start()
        await tf1.computation()
        tf2 = add(700, 800).start()
        return tf1.value, await tf2.task()

    assert asyncio.run(cell()) == (150, 1500)


def test_sync_run_does_not_run_on_caller_loop_thread():
    @delayed
    def ident():
        import threading

        return threading.get_ident()

    def body():
        return threading.get_ident(), ident().run()

    caller, worker_thread = in_running_loop(body)
    assert caller != worker_thread


def test_nested_multi_in_running_loop():
    """test_nested_transformations_multi, with the top level inside a running loop.

    This is the case that deadlocked the jupyter-sync branch (91bec26).
    """

    main_pid = os.getpid()
    job_count = 10

    @delayed
    def outer(label: str):
        from seamless.transformer import delayed, direct

        @direct
        def middle(label: str):
            from seamless.transformer import delayed

            def leaf(label: str):
                import os
                import time
                from seamless.transformer import global_lock

                with global_lock:
                    time.sleep(0.5)
                return label, os.getpid()

            leaf = delayed(leaf)
            left = leaf(f"{label}-a").start()
            right = leaf(f"{label}-b").start()
            return left.run(), right.run()

        return middle(f"{label}-1"), middle(f"{label}-2")

    def body():
        tasks = [outer(f"job-{idx}").start() for idx in range(job_count)]
        return [tf.run() for tf in tasks]

    start = time.perf_counter()
    results = in_running_loop(body)
    duration = time.perf_counter() - start

    expected_labels = {
        f"job-{idx}-{suffix}"
        for idx in range(job_count)
        for suffix in ("1-a", "1-b", "2-a", "2-b")
    }
    seen_labels = set()
    for outer_res in results:
        for mid in outer_res:
            for label, pid in mid:
                seen_labels.add(label)
                assert pid == main_pid
    assert seen_labels == expected_labels
    assert duration < 2.0 * job_count + 2


def test_cancel_async_waits_for_task_started_in_loop():
    @delayed
    def slow(a):
        import time

        time.sleep(3)
        return a

    async def cell():
        tf = slow(time.time()).start()
        await asyncio.sleep(0.5)
        task = tf._computation_task
        result = await tf.cancel_async()
        return result, tf.status, task.done()

    result, status, task_done = asyncio.run(cell())
    assert result is True
    assert status == "Status: canceled"
    assert task_done
