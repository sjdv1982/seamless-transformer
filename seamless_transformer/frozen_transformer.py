"""FrozenTransformer: an immutable copy of a Transformer's builder settings.

A FrozenTransformer is not a Transformation and not an owner. It holds the
builder's settings and pin arguments as they are, unconverted and without a
checksum, and it is the input of every path that needs a stable view of a
Transformer: building a Transformation (via a PreTransformation), cloning a
builder, and binding a standalone Transformer into a workflow Context.

It claims nothing on the checksums it references. A bound backend may attach
neutral hand-over leases (``leases``), which only bridge the gap until the
consumer acquires its own claims; they end when the FrozenTransformer is
consumed or garbage-collected. Keeping a FrozenTransformer alive is not a way
to keep its inputs alive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FrozenTransformer:
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
    input_celltypes: dict[str, str] = field(default_factory=dict)
    # None means a standalone call, whose direct arguments are literals.
    literal_pins: frozenset[str] | None = None
    # Optional neutral hand-over leases supplied by a bound backend.
    leases: tuple[Any, ...] = ()
    # Operational only: passed on to the built Transformation, never part of identity.
    streaming: bool = False


__all__ = ["FrozenTransformer"]
