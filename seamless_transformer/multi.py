from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, AsyncIterator, Callable, Iterable, Iterator

from .transformation_class import Transformation


class TransformationIterableBase:
    """
    Base class for an indexed list of Transformations with a completion callback.
    Supports full item access including slices by delegating to the embedded list.
    """

    def __init__(self, transformations: list[Transformation]):
        self._transformations = list(transformations)

    def __iter__(self) -> Iterator[Transformation]:
        return iter(self._transformations)

    def __len__(self) -> int:
        return len(self._transformations)

    def __getitem__(self, key):
        return self._transformations[key]

    def finish(self, index: int) -> None:
        """Called when transformation at position `index` has finished."""


class TransformationList(TransformationIterableBase):
    """
    Wrap a list of transformations with progress reporting and completion tracking.
    """

    def __init__(
        self,
        transformations: list[Transformation],
        show_progress: bool = True,
        on_error: str | None = None,
        store_details: bool = False,
    ):
        if on_error not in (None, "print", "raise"):
            raise ValueError("on_error must be None, 'print', or 'raise'")
        super().__init__(transformations)
        self._total = len(self._transformations)
        self._finished = 0
        self._errors = 0
        self._cancelled = 0
        self._show_progress = show_progress
        self._on_error = on_error
        self._store_details = store_details
        self._stored_exceptions: dict[int, str | None] = {}
        self._stored_checksums: dict[int, Any] = {}
        self._pbar = None
        self._error_pbar = None
        self._cancel_pbar = None
        if show_progress:
            import tqdm as _tqdm

            self._pbar = _tqdm.tqdm(total=self._total, desc="finished")

    def finish(self, index: int) -> None:
        tf = self._transformations[index]
        if tf is None:
            return
        self._finished += 1

        is_error = tf._exception is not None
        is_cancelled = (
            not is_error
            and tf._computation_task is not None
            and tf._computation_task.cancelled()
        )

        if is_error:
            self._errors += 1
            if self._on_error == "print":
                print(tf._exception, end="")
            elif self._on_error == "raise":
                raise RuntimeError(f"Transformation {index} failed:\n{tf._exception}")

        if is_cancelled:
            self._cancelled += 1

        if self._store_details:
            self._stored_exceptions[index] = tf._exception
            self._stored_checksums[index] = getattr(tf, "_result_checksum", None)
            self._transformations[index] = None

        if self._show_progress and self._pbar is not None:
            self._pbar.update(1)
            if is_error and self._errors == 1:
                import tqdm as _tqdm

                self._error_pbar = _tqdm.tqdm(
                    total=self._finished, desc="error", leave=False
                )
            if self._error_pbar is not None:
                self._error_pbar.total = self._finished
                self._error_pbar.update(1 if is_error else 0)
                self._error_pbar.refresh()
            if is_cancelled and self._cancelled == 1:
                import tqdm as _tqdm

                self._cancel_pbar = _tqdm.tqdm(
                    total=self._finished, desc="cancelled", leave=False
                )
            if self._cancel_pbar is not None:
                self._cancel_pbar.total = self._finished
                self._cancel_pbar.update(1 if is_cancelled else 0)
                self._cancel_pbar.refresh()

    def close(self) -> None:
        for pbar in (self._cancel_pbar, self._error_pbar, self._pbar):
            if pbar is not None:
                pbar.close()


async def _parallel_stream(
    indexed_items: list[tuple[int, Transformation]],
    nparallel: int,
    finish_cb: Callable[[int], None],
) -> AsyncIterator[Transformation]:
    """Keep at most nparallel transformations running and yield finished transformations in input order."""
    queue = deque(indexed_items)
    working: dict[asyncio.Task[Any], tuple[int, Transformation]] = {}
    completed: dict[int, Transformation] = {}
    next_index = 0
    try:
        while queue or working:
            while queue and len(working) < nparallel:
                idx, tf = queue.popleft()
                working[tf.task()] = (idx, tf)

            if not working:
                break

            done, _ = await asyncio.wait(
                list(working.keys()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                idx, tf = working.pop(task)
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                finish_cb(idx)
                completed[idx] = tf
            while next_index in completed:
                yield completed.pop(next_index)
                next_index += 1
    except BaseException:
        for task in working:
            task.cancel()
        if working:
            await asyncio.gather(*working.keys(), return_exceptions=True)
        raise


class _ParallelIterator:
    def __init__(
        self,
        indexed_items: list[tuple[int, Transformation]],
        nparallel: int,
        finish_cb: Callable[[int], None],
        close_cb: Callable[[], None],
    ):
        self._loop = asyncio.new_event_loop()
        self._agen = _parallel_stream(indexed_items, nparallel, finish_cb)
        self._close_cb = close_cb
        self._closed = False

    def __iter__(self) -> "_ParallelIterator":
        return self

    def __next__(self) -> Transformation:
        if self._closed:
            raise StopIteration
        try:
            return self._loop.run_until_complete(self._agen.__anext__())
        except StopAsyncIteration:
            self.close()
            raise StopIteration from None
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.run_until_complete(self._agen.aclose())
        finally:
            self._loop.close()
            self._close_cb()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _prepare_parallel(
    transformations: Iterable[Transformation],
    stage: str | None = None,
    substage: str | None = None,
) -> tuple[list[tuple[int, Transformation]], int, Callable[[int], None], Callable[[], None]]:
    if stage is not None or substage is not None:
        import seamless_config

        seamless_config.set_stage(stage, substage)

    from seamless_config.select import get_nparallel

    finish_cb = getattr(transformations, "finish", None)
    if not callable(finish_cb):
        finish_cb = lambda idx: None
    close_cb = getattr(transformations, "close", None)
    if not callable(close_cb):
        close_cb = lambda: None
    items = list(enumerate(transformations))
    return items, get_nparallel(), finish_cb, close_cb


def parallel(
    transformations: Iterable[Transformation],
    stage: str | None = None,
    substage: str | None = None,
) -> Iterator[Transformation]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "parallel() cannot be called from within a running event loop. "
            "Use parallel_async() instead."
        )

    items, nparallel, finish_cb, close_cb = _prepare_parallel(
        transformations, stage=stage, substage=substage
    )
    return _ParallelIterator(items, nparallel, finish_cb, close_cb)


async def parallel_async(
    transformations: Iterable[Transformation],
    stage: str | None = None,
    substage: str | None = None,
) -> AsyncIterator[Transformation]:
    items, nparallel, finish_cb, close_cb = _prepare_parallel(
        transformations, stage=stage, substage=substage
    )
    try:
        async for tf in _parallel_stream(items, nparallel, finish_cb):
            yield tf
    finally:
        close_cb()
