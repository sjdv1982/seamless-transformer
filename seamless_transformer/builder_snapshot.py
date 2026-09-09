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
    schema: str | None = None
    compilation: Any = None
    objects: Any = None
    header: str | None = None
    # Optional owning guards supplied by a bound backend during snapshot ingress.
    leases: tuple[Any, ...] = ()


__all__ = ["TransformerBuilderSnapshot"]
