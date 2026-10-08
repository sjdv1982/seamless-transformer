"""Forward child-process tqdm updates over the existing streaming channel."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import os
import sys
import threading
import time
from typing import Any, Callable

_LOGGER = logging.getLogger(__name__)
_PATCH_LOCK = threading.RLock()
_ACTIVE_PATCHES: list["_TqdmPatch"] = []
_TQDM_HIERARCHY_IMPORT_DEPTH = 0
_TQDM_IMPORT_DEPTH = 0
_DEVNULL = open(os.devnull, "w", encoding="utf-8")
_PATCHED_MODULES = ("tqdm", "tqdm.std", "tqdm.auto", "tqdm.asyncio")


def _active_patch() -> "_TqdmPatch | None":
    with _PATCH_LOCK:
        return _ACTIVE_PATCHES[-1] if _ACTIVE_PATCHES else None


def _plain_value(value: Any, *, limit: int = 1024) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > limit:
            return value[:limit]
        return value
    try:
        return str(value)[:limit]
    except Exception:
        return None


class _PatchLoader(importlib.abc.Loader):
    """Run the normal tqdm loader, then replace its public class aliases."""

    def __init__(self, loader: Any, fullname: str) -> None:
        self._loader = loader
        self._fullname = fullname

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module) -> None:
        global _TQDM_HIERARCHY_IMPORT_DEPTH, _TQDM_IMPORT_DEPTH
        restored = []
        is_hierarchy_module = self._fullname in {"tqdm.auto", "tqdm.asyncio"}
        is_package = self._fullname == "tqdm"
        with _PATCH_LOCK:
            # These modules derive classes from tqdm.std. Let them see the
            # original class hierarchy while they define those subclasses.
            if is_hierarchy_module:
                _TQDM_HIERARCHY_IMPORT_DEPTH += 1
                for patch in reversed(tuple(_ACTIVE_PATCHES)):
                    for target, attr, wrapper, original in reversed(patch._originals):
                        if getattr(target, attr, None) is wrapper:
                            restored.append((target, attr, wrapper, original))
                            setattr(target, attr, original)
            if is_package:
                _TQDM_IMPORT_DEPTH += 1
        try:
            self._loader.exec_module(module)
        finally:
            with _PATCH_LOCK:
                for target, attr, wrapper, original in reversed(restored):
                    if getattr(target, attr, None) is original:
                        setattr(target, attr, wrapper)
                if is_hierarchy_module:
                    _TQDM_HIERARCHY_IMPORT_DEPTH -= 1
                if is_package:
                    _TQDM_IMPORT_DEPTH -= 1
                if (
                    _TQDM_HIERARCHY_IMPORT_DEPTH == 0
                    and _TQDM_IMPORT_DEPTH == 0
                ):
                    for patch in tuple(_ACTIVE_PATCHES):
                        patch._patch_loaded_modules()


class _TqdmPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _PATCHED_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchLoader(spec.loader, fullname)
        return spec


class _TqdmPatch:
    def __init__(
        self,
        notifier: Callable[[dict[str, Any]], Any],
        *,
        min_interval: float = 2.0,
        before_update: Callable[[bool, Any], bool] | None = None,
        capture_watermark: Callable[[], Any] | None = None,
    ) -> None:
        self.notifier = notifier
        self._min_interval = self._bounded_interval(min_interval)
        self._before_update = before_update
        self._capture_watermark = capture_watermark
        self._bars: set[Any] = set()
        self._lock = threading.RLock()
        self._finder = _TqdmPatchFinder()
        self._originals: list[tuple[Any, str, Any, Any]] = []
        self._entered = False
        self._closed = False

    @staticmethod
    def _bounded_interval(value: Any) -> float:
        try:
            return min(max(float(value), 0.0), 60.0)
        except (TypeError, ValueError, OverflowError):
            return 2.0

    def __enter__(self) -> "_TqdmPatch":
        with _PATCH_LOCK:
            if self._entered:
                return self
            self._entered = True
            self._closed = False
            _ACTIVE_PATCHES.append(self)
            sys.meta_path.insert(0, self._finder)
            self._patch_loaded_modules()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        with self._lock:
            bars = tuple(self._bars)
        for bar in bars:
            try:
                bar.close()
            except Exception:
                _LOGGER.debug("Could not close a streamed tqdm bar", exc_info=True)

        with _PATCH_LOCK:
            self._closed = True
            try:
                sys.meta_path.remove(self._finder)
            except ValueError:
                pass
            for module, attr, wrapper, original in reversed(self._originals):
                if getattr(module, attr, None) is wrapper:
                    setattr(module, attr, original)
            self._originals.clear()
            try:
                _ACTIVE_PATCHES.remove(self)
            except ValueError:
                pass
            self._entered = False

    def _patch_loaded_modules(self) -> None:
        for module_name in _PATCHED_MODULES:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            original = getattr(module, "tqdm", None)
            if not isinstance(original, type):
                continue
            if getattr(original, "_seamless_stream_tqdm", False):
                continue
            wrapper = self._make_subclass(original)
            self._originals.append((module, "tqdm", wrapper, original))
            setattr(module, "tqdm", wrapper)

    def _make_subclass(self, original):
        patch_type = type(self)

        class StreamingTqdm(original):
            _seamless_stream_tqdm = True

            def __init__(self, *args, **kwargs):
                self._stream_patch = _active_patch()
                self._stream_bar_id = f"{os.getpid()}-{id(self)}"
                self._stream_lock = threading.RLock()
                self._stream_send_lock = threading.Lock()
                self._stream_last_emit = time.monotonic()
                self._stream_pending: dict[str, Any] | None = None
                self._stream_pending_watermark: Any = None
                self._stream_pending_waiting_text = False
                self._stream_ready = False
                self._stream_in_update = False
                self._stream_closing = False
                self._stream_closed = False

                if self._stream_patch is not None:
                    args = list(args)
                    if len(args) > 4:
                        args[4] = _DEVNULL
                    else:
                        kwargs["file"] = _DEVNULL
                    if len(args) > 10:
                        args[10] = bool(args[10])
                    else:
                        requested_disable = kwargs.get("disable")
                        kwargs["disable"] = (
                            bool(requested_disable)
                            if requested_disable is not None
                            else False
                        )
                super().__init__(*args, **kwargs)

                if self._stream_patch is not None and not self.disable:
                    self._stream_ready = True
                    self._stream_patch._register_bar(self)
                    self._send(
                        {
                            "kind": "tqdm_open",
                            "bar_id": self._stream_bar_id,
                            "desc": _plain_value(getattr(self, "desc", None)),
                            "total": _plain_value(getattr(self, "total", None)),
                            "unit": _plain_value(getattr(self, "unit", "it")),
                            "unit_scale": _plain_value(
                                getattr(self, "unit_scale", False)
                            ),
                            "mininterval": _plain_value(
                                getattr(self, "mininterval", 0.1)
                            ),
                            "bar_format": _plain_value(
                                getattr(self, "bar_format", None)
                            ),
                            "n": _plain_value(getattr(self, "n", 0)),
                        }
                    )
                    self._queue_update(force=True)

            def update(self, n=1):
                self._stream_in_update = True
                try:
                    result = super().update(n)
                finally:
                    self._stream_in_update = False
                if self._stream_ready and not self._stream_closing:
                    self._queue_update()
                return result

            def refresh(self, nolock=False, lock_args=None):
                if self._stream_ready and not self._stream_closing:
                    if self._stream_in_update:
                        return False
                    return self._queue_update()
                return super().refresh(nolock=nolock, lock_args=lock_args)

            def display(self, msg=None, pos=None):
                if self._stream_patch is not None:
                    return True
                return super().display(msg=msg, pos=pos)

            def close(self):
                if self._stream_closed:
                    return
                if self._stream_patch is None or not self._stream_ready:
                    return super().close()
                with self._stream_lock:
                    if self._stream_closing or self._stream_closed:
                        return
                    self._stream_closing = True
                try:
                    super().close()
                finally:
                    with self._stream_send_lock:
                        before_update = self._stream_patch._before_update
                        if before_update is not None:
                            before_update(True, self._stream_pending_watermark)
                        with self._stream_lock:
                            final_update = self._make_update()
                            self._stream_pending = None
                            self._stream_pending_watermark = None
                            self._stream_pending_waiting_text = False
                            self._stream_closed = True
                        self._notify(final_update)
                        self._notify(
                            {
                                "kind": "tqdm_close",
                                "bar_id": self._stream_bar_id,
                                "n": _plain_value(getattr(self, "n", 0)),
                                "total": _plain_value(getattr(self, "total", None)),
                            }
                        )
                    self._stream_patch._unregister_bar(self)

            def _update_stream_state(self, *, min_interval: float) -> None:
                with self._stream_lock:
                    self._stream_patch._min_interval = patch_type._bounded_interval(
                        min_interval
                    )
                    pending = self._stream_pending
                if pending is not None:
                    self._queue_update()

            def _flush_stream_pending(self) -> None:
                with self._stream_lock:
                    pending = self._stream_pending is not None
                if pending:
                    self._queue_update()

            def _make_update(self) -> dict[str, Any]:
                try:
                    format_dict = self.format_dict
                except Exception:
                    format_dict = {}
                postfix = getattr(self, "postfix", None)
                if isinstance(postfix, dict):
                    postfix = ", ".join(f"{key}={value}" for key, value in postfix.items())
                return {
                    "kind": "tqdm_update",
                    "bar_id": self._stream_bar_id,
                    "n": _plain_value(getattr(self, "n", 0)),
                    "total": _plain_value(getattr(self, "total", None)),
                    "elapsed": _plain_value(format_dict.get("elapsed")),
                    "rate": _plain_value(format_dict.get("rate")),
                    "postfix": _plain_value(postfix),
                }

            def _queue_update(self, *, force: bool = False) -> bool:
                if (
                    not self._stream_ready
                    or self._stream_closing
                    or self._stream_closed
                ):
                    return False
                patch = self._stream_patch
                if patch is None or patch._closed:
                    return False
                with self._stream_send_lock:
                    with self._stream_lock:
                        if self._stream_closing or self._stream_closed:
                            return False
                        if (
                            self._stream_pending is None
                            or not self._stream_pending_waiting_text
                        ):
                            capture_watermark = patch._capture_watermark
                            self._stream_pending_watermark = (
                                capture_watermark()
                                if capture_watermark is not None
                                else None
                            )
                        self._stream_pending = self._make_update()
                        now = time.monotonic()
                        if (
                            not force
                            and now - self._stream_last_emit < patch._min_interval
                        ):
                            return False
                    before_update = patch._before_update
                    if before_update is not None and not before_update(
                        force, self._stream_pending_watermark
                    ):
                        with self._stream_lock:
                            self._stream_pending_waiting_text = True
                        return False
                    now = time.monotonic()
                    with self._stream_lock:
                        if self._stream_closing or self._stream_closed:
                            return False
                        message = self._stream_pending
                        self._stream_pending = None
                        self._stream_pending_watermark = None
                        self._stream_pending_waiting_text = False
                        self._stream_last_emit = now
                    self._notify(message)
                return True

            def _send(self, message: dict[str, Any]) -> None:
                with self._stream_send_lock:
                    self._notify(message)

            def _notify(self, message: dict[str, Any]) -> None:
                patch = self._stream_patch
                if patch is None or patch._closed:
                    return
                try:
                    patch.notifier(message)
                except Exception:
                    _LOGGER.debug("Failed to send streamed tqdm state", exc_info=True)

        StreamingTqdm.__name__ = f"Streaming{original.__name__}"
        StreamingTqdm.__qualname__ = StreamingTqdm.__name__
        StreamingTqdm.__module__ = original.__module__
        return StreamingTqdm

    def _register_bar(self, bar: Any) -> None:
        with self._lock:
            if not self._closed:
                self._bars.add(bar)

    def _unregister_bar(self, bar: Any) -> None:
        with self._lock:
            self._bars.discard(bar)

    def update_throttle(self, *, min_interval: float) -> None:
        with self._lock:
            if self._closed:
                return
            self._min_interval = self._bounded_interval(min_interval)
            bars = tuple(self._bars)
        for bar in bars:
            bar._update_stream_state(min_interval=self._min_interval)

    def flush_pending(self) -> None:
        with self._lock:
            if self._closed:
                return
            bars = tuple(self._bars)
        for bar in bars:
            bar._flush_stream_pending()


def install_tqdm_patch(
    notifier: Callable[[dict[str, Any]], Any],
    *,
    min_interval: float = 2.0,
    before_update: Callable[[bool, Any], bool] | None = None,
    capture_watermark: Callable[[], Any] | None = None,
) -> _TqdmPatch:
    """Patch public tqdm aliases for a streaming transformation request."""
    return _TqdmPatch(
        notifier,
        min_interval=min_interval,
        before_update=before_update,
        capture_watermark=capture_watermark,
    )


def update_active_tqdm_throttles(*, min_interval: float) -> None:
    """Apply a parent throttle update to tqdm bars active in this child."""
    with _PATCH_LOCK:
        patches = tuple(_ACTIVE_PATCHES)
    for patch in patches:
        patch.update_throttle(min_interval=min_interval)


def flush_active_tqdm_updates() -> None:
    """Emit coalesced progress states once their throttle interval elapses."""
    with _PATCH_LOCK:
        patches = tuple(_ACTIVE_PATCHES)
    for patch in patches:
        patch.flush_pending()
