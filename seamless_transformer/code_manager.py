"""Utilities to manage syntactic and semantic transformer code."""

from __future__ import annotations

import ast
from typing import Dict, Iterable, Tuple

from seamless import Buffer, Checksum

from .getsource import ast_dump


def _coerce_checksum(value) -> Checksum:
    """Return a Checksum instance from hex / bytes / Checksum input."""
    if isinstance(value, Checksum):
        return value
    return Checksum(value)


class CodeManager:
    """Keeps track of transformer code buffers and their semantic equivalents."""

    def __init__(self) -> None:
        # map syntactic checksum hex -> semantic Checksum
        self._syntactic_to_semantic: Dict[str, Checksum] = {}
        # map semantic checksum hex -> set of syntactic checksum hex strings
        self._semantic_to_syntactic: Dict[str, set[str]] = {}
        # normal reference counts
        self._syntactic_direct_refs: Dict[str, int] = {}
        self._semantic_direct_refs: Dict[str, int] = {}
        # guard reference counts (semantic guards syntactic, syntactic guards semantic)
        self._syntactic_guard_refs: Dict[str, int] = {}
        self._semantic_guard_refs: Dict[str, int] = {}
        # keep semantic buffers alive so they remain resolvable across processes
        self._semantic_buffers: Dict[str, Buffer] = {}
        self._refholds_released = False
        from seamless.reference_lifecycle import register_refholder

        register_refholder(self)

    # --- registration helpers -------------------------------------------------
    def track_code_buffer(self, code_buffer: Buffer) -> Tuple[Checksum, Checksum]:
        """Register a transformer code buffer and return (semantic, syntactic) checksums."""
        if not isinstance(code_buffer, Buffer):
            code_buffer = Buffer(code_buffer, "python")
        syntactic_checksum = code_buffer.get_checksum()
        syntactic_hex = syntactic_checksum.hex()

        semantic_checksum = self._syntactic_to_semantic.get(syntactic_hex)
        if semantic_checksum is None:
            semantic_checksum = self._calculate_semantic(code_buffer)
            semantic_hex = semantic_checksum.hex()
            self._syntactic_to_semantic[syntactic_hex] = semantic_checksum
            self._semantic_to_syntactic.setdefault(semantic_hex, set()).add(
                syntactic_hex
            )
        return semantic_checksum, syntactic_checksum

    def _calculate_semantic(self, code_buffer: Buffer) -> Checksum:
        """Calculate the semantic checksum for a python code buffer."""
        code_text = code_buffer.decode()
        tree = ast.parse(code_text, filename="<transformer>")
        semantic_dump = ast_dump(tree)
        semantic_buffer = Buffer(semantic_dump, "text")
        checksum = semantic_buffer.get_checksum()
        # Keep the semantic buffer strongly referenced so child workers can download it.
        self._semantic_buffers[checksum.hex()] = semantic_buffer
        return checksum

    # --- reference counting ---------------------------------------------------
    def incref_semantic(self, checksum) -> None:
        """Increment semantic reference count and guard syntactic counterparts."""
        checksum = _coerce_checksum(checksum)
        key = checksum.hex()
        self._semantic_direct_refs[key] = self._semantic_direct_refs.get(key, 0) + 1
        checksum.incref_refholder()
        self._enable_syntactic_guards(self._semantic_to_syntactic.get(key, ()))

    def decref_semantic(self, checksum) -> None:
        """Decrement semantic reference count and release guards if needed."""
        checksum = _coerce_checksum(checksum)
        key = checksum.hex()
        current = self._semantic_direct_refs.get(key)
        if current is None:
            raise KeyError(key)
        if current <= 1:
            self._semantic_direct_refs.pop(key, None)
            self._disable_syntactic_guards(self._semantic_to_syntactic.get(key, ()))
            checksum.decref_refholder()
        else:
            self._semantic_direct_refs[key] = current - 1

    def incref_syntactic(self, checksum) -> None:
        """Increment syntactic reference count and guard the semantic checksum."""
        checksum = _coerce_checksum(checksum)
        key = checksum.hex()
        self._syntactic_direct_refs[key] = self._syntactic_direct_refs.get(key, 0) + 1
        checksum.incref_refholder()

        semantic_checksum = self._syntactic_to_semantic.get(key)
        if semantic_checksum is None:
            return
        sem_key = semantic_checksum.hex()
        self._semantic_guard_refs[sem_key] = (
            self._semantic_guard_refs.get(sem_key, 0) + 1
        )
        semantic_checksum.incref_refholder()

    def decref_syntactic(self, checksum) -> None:
        """Decrement syntactic reference count and release semantic guard if needed."""
        checksum = _coerce_checksum(checksum)
        key = checksum.hex()
        current = self._syntactic_direct_refs.get(key)
        if current is None:
            raise KeyError(key)
        if current <= 1:
            self._syntactic_direct_refs.pop(key, None)
        else:
            self._syntactic_direct_refs[key] = current - 1
        checksum.decref_refholder()

        semantic_checksum = self._syntactic_to_semantic.get(key)
        if semantic_checksum is None:
            return
        sem_key = semantic_checksum.hex()
        guard_refs = self._semantic_guard_refs.get(sem_key, 0)
        if guard_refs <= 1:
            self._semantic_guard_refs.pop(sem_key, None)
        else:
            self._semantic_guard_refs[sem_key] = guard_refs - 1
        semantic_checksum.decref_refholder()

    # --- guard helpers --------------------------------------------------------
    def _enable_syntactic_guards(self, syntactic_keys: Iterable[str]) -> None:
        for syn_key in syntactic_keys:
            if self._syntactic_guard_refs.get(syn_key):
                continue
            self._syntactic_guard_refs[syn_key] = 1
            checksum = _coerce_checksum(syn_key)
            checksum.incref_refholder()

    def _disable_syntactic_guards(self, syntactic_keys: Iterable[str]) -> None:
        for syn_key in syntactic_keys:
            if syn_key not in self._syntactic_guard_refs:
                continue
            self._syntactic_guard_refs.pop(syn_key, None)
            checksum = _coerce_checksum(syn_key)
            checksum.decref_refholder()

    def _refheld_checksums(self):
        if self._refholds_released:
            return ()
        claims = []
        for key, count in self._syntactic_direct_refs.items():
            checksum = Checksum(key)
            claims.extend((checksum, "syntactic:direct") for _ in range(count))
        for key, count in self._syntactic_guard_refs.items():
            checksum = Checksum(key)
            claims.extend((checksum, "syntactic:guard") for _ in range(count))
        for key, count in self._semantic_direct_refs.items():
            checksum = Checksum(key)
            claims.extend((checksum, "semantic:direct") for _ in range(count))
        for key, count in self._semantic_guard_refs.items():
            checksum = Checksum(key)
            claims.extend((checksum, "semantic:guard") for _ in range(count))
        return claims

    def _release_refholds(self) -> None:
        if self._refholds_released:
            return
        for checksum, _role in list(self._refheld_checksums()):
            checksum.decref_refholder()
        self._syntactic_direct_refs.clear()
        self._syntactic_guard_refs.clear()
        self._semantic_direct_refs.clear()
        self._semantic_guard_refs.clear()
        self._semantic_buffers.clear()
        self._refholds_released = True

    def __del__(self):
        try:
            self._release_refholds()
        except Exception:
            pass

    def get_syntactic_checksums(self, semantic_checksum) -> list[Checksum]:
        """Return syntactic checksums mapped to a semantic checksum."""
        checksum = _coerce_checksum(semantic_checksum)
        syn_hex_values = self._semantic_to_syntactic.get(checksum.hex(), set())
        return [Checksum(hex_value) for hex_value in syn_hex_values]


_CODE_MANAGER: CodeManager | None = None


def get_code_manager() -> CodeManager:
    """Return the process-wide code manager instance."""
    global _CODE_MANAGER
    if _CODE_MANAGER is None:
        _CODE_MANAGER = CodeManager()
    return _CODE_MANAGER


__all__ = ["CodeManager", "get_code_manager"]
