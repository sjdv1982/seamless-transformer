"""Compiled input contracts, shared by builders and native executors.

Pre-hash validation never resolves an input checksum. Full validation uses only
an explicitly supplied buffer; literals and executors supply that buffer.
"""

import warnings
from collections.abc import Mapping

import numpy as np
from seamless.checksum.hash_type import get_hash_type
from seamless.checksum.null import canonicalize_checksum, is_null, is_null_value

from seamless import Buffer, Checksum


class CompiledPinCelltypeError(TypeError):
    pass


class CompiledPinCelltypeWarning(UserWarning):
    pass


class CompiledMixedValueError(TypeError):
    pass


class CompiledPinSchemaError(TypeError):
    pass


DECLARATIONS = {
    "integer": ("int", "binary", "mixed"),
    "floating": ("float", "binary", "mixed"),
    "boolean": ("bool", "binary", "mixed"),
    "binary": ("binary", "mixed"),
    "char": ("bytes", "binary"),
    "char-array": ("bytes", "text", "binary"),
}
SCHEMA_CELLTYPES = {
    "integer": "int",
    "floating": "float",
    "boolean": "bool",
    "binary": "binary",
    "char": None,
    "char-array": None,
}
ALLOWED_CELLTYPES = frozenset(c for values in DECLARATIONS.values() for c in values)


def parameter_kind(parameter):
    name = getattr(parameter.dtype, "name", None)
    if name == "char" and parameter.shape is None:
        return "char"
    if name == "char" and len(parameter.shape) == 1:
        return "char-array"
    if parameter.shape is not None or name is None:
        return "binary"
    if name.startswith(("int", "uint")):
        return "integer"
    return {"bool": "boolean", "float32": "floating", "float64": "floating"}.get(
        name, "binary"
    )


def describe(parameter, celltype):
    return (
        f"Compiled pin {parameter.name!r} declares celltype {celltype!r}, "
        f"schema dtype {getattr(parameter.dtype, 'name', parameter.dtype)!r}, "
        f"shape {parameter.shape}"
    )


def validate_declarations(sig, celltypes, *, warn=False):
    errors = []
    for p in sig.inputs:
        kind = parameter_kind(p)
        ct = celltypes.get(p.name, "mixed")
        if ct not in DECLARATIONS[kind]:
            reason = (
                " needs an explicit celltype; mixed (auto) is unavailable."
                if ct == "mixed"
                else " is incompatible."
            )
            errors.append(
                describe(p, ct)
                + reason
                + f" Schema celltype: {SCHEMA_CELLTYPES[kind]!r}. Allowed pin celltypes: {DECLARATIONS[kind]}."
            )
    if errors:
        message = "\n".join(errors)
        if warn:
            warnings.warn(message, CompiledPinCelltypeWarning, stacklevel=3)
        else:
            raise CompiledPinCelltypeError(message)


class SchemaCelltypesView(Mapping):
    """Read-only derived types with declaration diagnostics in inspection."""

    def __init__(self, signature, declarations):
        self._types = (
            {p.name: SCHEMA_CELLTYPES[parameter_kind(p)] for p in signature.inputs}
            if signature
            else {}
        )
        self._diagnostic = None
        if signature is not None:
            try:
                validate_declarations(signature, declarations)
            except CompiledPinCelltypeError as exc:
                self._diagnostic = str(exc)

    def __getitem__(self, key):
        return self._types[key]

    def __iter__(self):
        return iter(self._types)

    def __len__(self):
        return len(self._types)

    def __repr__(self):
        diagnostic = f"; incompatible: {self._diagnostic}" if self._diagnostic else ""
        return f"SchemaCelltypes({self._types!r}{diagnostic})"


def validate_stage1(schema, celltypes, optional_pins=(), metavars=None):
    import yaml
    from seamless_signature import Signature, generate_header

    if not schema:
        raise ValueError("compiled transformer schema is not set")
    sig = Signature.from_dict(yaml.safe_load(schema))
    generate_header(sig)
    if optional_pins:
        raise CompiledPinCelltypeError("compiled inputs cannot be optional")
    validate_declarations(sig, celltypes)
    missing = [
        f"max{w}" for w in sig.output_wildcards if f"max{w}" not in (metavars or {})
    ]
    if missing:
        raise ValueError(f"compiled transformer metavars are incomplete: {missing}")
    return sig


def validate_pin(parameter, celltype, checksum, *, buffer=None, wildcards=None):
    """Validate checksum facts, and (only if supplied) its serialized value."""
    from .run import _numpy_dtype

    prefix = describe(parameter, celltype)

    def fail(message):
        raise CompiledPinSchemaError(prefix + ": " + message)

    def mixed_fail():
        raise CompiledMixedValueError(
            prefix
            + ": JSON text/containers and proper mixed values have no native ABI. "
            f"Allowed declarations: {DECLARATIONS[parameter_kind(parameter)]}"
        )

    if checksum is None or not Checksum(checksum):
        fail("missing input checksum")
    checksum = canonicalize_checksum(Checksum(checksum), celltype)
    if is_null_value(checksum):
        if celltype != "bytes":
            fail("null is not a native input")
        if not is_null(checksum):
            fail("ambiguous checksum: 4-byte content 'null' or non-canonical JSON null")
        buffer = Buffer(b"null\n")
    if buffer is None:
        from hashlib import sha256

        # These buffers are mathematical checksum facts, not cache lookups.
        for raw in (
            b"true",
            b"true\n",
            b"false",
            b"false\n",
            b"{}",
            b"{}\n",
            b"[]",
            b"[]\n",
            b'""',
            b'""\n',
        ):
            if checksum.hex() == sha256(raw).hexdigest():
                buffer = Buffer(raw)
                break
    ht = get_hash_type(checksum)
    if buffer is not None and (ht is None or ht.is_untested):
        from seamless.checksum.hash_type import register_hash_type_for_buffer

        try:
            ht = register_hash_type_for_buffer(checksum, buffer)
        except (TypeError, ValueError, UnicodeError, AssertionError):
            fail(f"checksum is not readable as {celltype}")
    expected = _numpy_dtype(parameter.dtype)
    expected_rank = 0 if parameter.shape is None else len(parameter.shape)
    if ht is not None:
        if ht.deserializable_as(celltype, checksum=checksum) is False:
            fail(f"checksum is not readable as {celltype}")
        if celltype == "mixed" and ht.kind.name in (
            "JSON_OBJECT",
            "JSON_ARRAY",
            "MIXED_OBJECT",
            "MIXED_ARRAY",
        ):
            mixed_fail()
        if celltype in ("mixed", "binary") and ht.kind.name == "NUMPY":
            rank = int(ht.rank)
            if (rank < 3 and rank != expected_rank) or (
                rank == 3 and expected_rank < 3
            ):
                fail("rank does not match schema")
            expected_class = (
                "STRUCTURED"
                if expected.fields
                else ("NUMERIC" if expected.kind in "iufbc" else "NONNUMERIC")
            )
            if ht.dtype.name != expected_class:
                fail("dtype class does not match schema")
    if buffer is None:
        return
    if buffer.content.startswith(b"\x93NUMPY") and celltype in ("mixed", "binary"):
        from seamless.util.mixed.io.from_stream import parse_npy_header

        try:
            _, _, dtype0, _ = parse_npy_header(buffer.content)
        except (TypeError, ValueError, EOFError):
            fail(f"checksum is not readable as {celltype}")
        expected0 = _numpy_dtype(parameter.dtype)
        if (parameter.shape is not None or expected0.kind in "VS") and (
            not dtype0.isnative or dtype0 != expected0
        ):
            fail(f"dtype is {dtype0}, expected native {expected0}")
    try:
        value = buffer.get_value(celltype)
    except (TypeError, ValueError, UnicodeError, AssertionError):
        fail(f"checksum is not readable as {celltype}")
    if value is None:
        fail("null is not a native input")
    if celltype == "mixed" and isinstance(value, (str, list, tuple, dict)):
        mixed_fail()
    expected = _numpy_dtype(parameter.dtype)
    shape = parameter.shape
    if celltype in ("bytes", "text"):
        if isinstance(value, Buffer):
            value = value.content
        if isinstance(value, str):
            value = value.encode("utf8")
        if shape is None:
            if len(value) != 1:
                fail("scalar char requires exactly one byte")
            return value[0]
        actual_shape = (len(value),)
    else:
        array = np.asarray(value)
        actual_shape = array.shape
        if shape is None and expected.kind in "iufbc":
            if actual_shape != ():
                fail("rank does not match schema")
            kind = array.dtype.kind
            allowed = {"i": "iu", "u": "iu", "f": "iuf", "b": "b", "c": "c"}[
                expected.kind
            ]
            if kind not in allowed:
                fail(f"expected scalar kind {allowed}, got {kind}")
            scalar = array[()]
            if expected.kind in "iu":
                bounds = np.iinfo(expected)
                if not bounds.min <= scalar <= bounds.max:
                    fail(f"value outside {expected} range")
            elif expected.kind in "fc":
                bounds = np.finfo(expected)
                parts = (
                    (scalar.real, scalar.imag) if expected.kind == "c" else (scalar,)
                )
                if any(np.isfinite(part) and abs(part) > bounds.max for part in parts):
                    fail(f"finite value outside {expected} range")
                if kind in "iu":
                    converted = expected.type(scalar)
                    if not np.isfinite(converted) or int(converted) != int(scalar):
                        fail(f"integer is not exactly representable as {expected}")
            return scalar.item()
        if array.dtype != expected or not array.dtype.isnative:
            fail(f"dtype is {array.dtype}, expected native {expected}")
        if not array.flags.aligned:
            fail("array is not aligned")
        value = array
    required_shape = () if shape is None else shape
    if len(actual_shape) != len(required_shape):
        fail("rank does not match schema")
    for actual, required in zip(actual_shape, required_shape):
        if isinstance(required, int):
            if actual != required:
                fail(f"dimension is {actual}, expected {required}")
        elif wildcards is not None:
            if actual > np.iinfo(np.uint32).max:
                fail(f"wildcard {required!r} exceeds unsigned int range")
            if required in wildcards and wildcards[required] != actual:
                fail(f"wildcard {required!r} has inconsistent sizes")
            wildcards[required] = actual
    if shape is None and expected.kind == "S":
        return value.tobytes()[0]
    return value


def validate_prepared(sig, prepared):
    wildcards = {}
    for p in sig.inputs:
        if p.name not in prepared:
            raise CompiledPinSchemaError(f"Compiled pin {p.name!r}: missing input")
        ct, _, cs = prepared[p.name]
        validate_pin(p, ct, cs, wildcards=wildcards)
