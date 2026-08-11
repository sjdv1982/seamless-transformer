"""Immutable transformer builder state exchanged with bound backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransformerBuilderSnapshot:
    codebuf: Any
    language: str
    celltypes: dict[str, str]
    optional_pins: frozenset[str]
    args: dict[str, Any]
    modules: dict[str, Any]
    globals: dict[str, Any]
    meta: dict[str, Any]
    environment: Any
    scratch: bool
    direct_print: bool
    local: bool | None
    call_mode: str
    callable: Any = None
    signature: Any = None


__all__ = ["TransformerBuilderSnapshot"]
