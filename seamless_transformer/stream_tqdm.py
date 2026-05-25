"""Temporary tqdm patching for streamed transformations."""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Callable


@contextlib.contextmanager
def install_tqdm_patch(notifier: Callable[[dict[str, Any]], None]):
    try:
        import tqdm as tqdm_module
        import tqdm.std as tqdm_std
    except Exception:
        yield
        return

    original_std = getattr(tqdm_std, "tqdm", None)
    original_root = getattr(tqdm_module, "tqdm", None)
    try:
        import tqdm.auto as tqdm_auto
    except Exception:
        tqdm_auto = None
        original_auto = None
    else:
        original_auto = getattr(tqdm_auto, "tqdm", None)

    if original_std is None:
        yield
        return

    devnull = open(os.devnull, "w")

    class StreamingTqdm(original_std):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["file"] = devnull
            super().__init__(*args, **kwargs)
            self._seamless_bar_id = f"{os.getpid()}-{id(self)}"
            self._seamless_last_emit = 0.0
            notifier(
                {
                    "kind": "tqdm_open",
                    "bar_id": self._seamless_bar_id,
                    "desc": getattr(self, "desc", None),
                    "total": getattr(self, "total", None),
                    "unit": getattr(self, "unit", None),
                    "n": getattr(self, "n", 0),
                }
            )

        def update(self, n=1):
            result = super().update(n)
            self._seamless_emit_update()
            return result

        def refresh(self, *args, **kwargs):
            result = super().refresh(*args, **kwargs)
            self._seamless_emit_update()
            return result

        def close(self):
            if not hasattr(self, "_seamless_bar_id"):
                return super().close()
            if getattr(self, "_seamless_closed", False):
                return super().close()
            self._seamless_closed = True
            notifier(
                {
                    "kind": "tqdm_close",
                    "bar_id": self._seamless_bar_id,
                    "n": getattr(self, "n", 0),
                    "total": getattr(self, "total", None),
                }
            )
            return super().close()

        def _seamless_emit_update(self) -> None:
            if not hasattr(self, "_seamless_bar_id"):
                return
            now = time.time()
            if now - getattr(self, "_seamless_last_emit", 0.0) < 0.5:
                return
            self._seamless_last_emit = now
            notifier(
                {
                    "kind": "tqdm_update",
                    "bar_id": self._seamless_bar_id,
                    "n": getattr(self, "n", 0),
                    "total": getattr(self, "total", None),
                    "elapsed": getattr(self, "format_dict", {}).get("elapsed"),
                    "rate": getattr(self, "format_dict", {}).get("rate"),
                }
            )

    tqdm_std.tqdm = StreamingTqdm
    tqdm_module.tqdm = StreamingTqdm
    if tqdm_auto is not None:
        tqdm_auto.tqdm = StreamingTqdm
    try:
        yield
    finally:
        tqdm_std.tqdm = original_std
        tqdm_module.tqdm = original_root
        if tqdm_auto is not None:
            tqdm_auto.tqdm = original_auto
        devnull.close()
