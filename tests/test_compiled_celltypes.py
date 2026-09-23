"""Compiled contracts are scoped to pins, not to upstream workflow data."""

from hashlib import sha256

import numpy as np
import pytest
import yaml
from seamless_signature import Signature
from seamless_transformer import Transformer
from seamless_transformer.compiled_validation import (
    DECLARATIONS,
    SCHEMA_CELLTYPES,
    CompiledMixedValueError,
    CompiledPinCelltypeError,
    CompiledPinCelltypeWarning,
    CompiledPinSchemaError,
    parameter_kind,
    validate_pin,
)

from seamless import Buffer, Cell, Checksum


def make(dtype="int32", shape=None, celltype="mixed", direct=False):
    tf = Transformer("c", compiled=True, direct=direct)
    inp = {"name": "x", "dtype": dtype}
    if shape is not None:
        inp["shape"] = shape
    schema = yaml.safe_dump(
        {"inputs": [inp], "outputs": [{"name": "result", "dtype": "int32"}]}
    )
    if dtype == "char" and (shape is None or len(shape) == 1):
        with pytest.warns(CompiledPinCelltypeWarning):
            tf.schema = schema
    else:
        tf.schema = schema
    tf.celltypes.x = celltype
    tf.code = "#include <stdint.h>\nint transform(int32_t x, int32_t *result) {*result=x; return 0;}"
    return tf


@pytest.mark.parametrize(
    "dtype,shape,derived,allowed",
    [
        ("int32", None, "int", ("int", "binary", "mixed")),
        ("uint64", None, "int", ("int", "binary", "mixed")),
        ("float32", None, "float", ("float", "binary", "mixed")),
        ("bool", None, "bool", ("bool", "binary", "mixed")),
        ("complex64", None, "binary", ("binary", "mixed")),
        ("float64", ["N"], "binary", ("binary", "mixed")),
        ("char", None, None, ("bytes", "binary")),
        ("char", ["N"], None, ("bytes", "text", "binary")),
        ("char", ["N", 4], "binary", ("binary", "mixed")),
    ],
)
def test_tables(dtype, shape, derived, allowed):
    p = {"name": "x", "dtype": dtype}
    if shape is not None:
        p["shape"] = shape
    sig = Signature.from_dict(
        {"inputs": [p], "outputs": [{"name": "result", "dtype": "int32"}]}
    )
    kind = parameter_kind(sig.inputs[0])
    assert SCHEMA_CELLTYPES[kind] == derived
    assert DECLARATIONS[kind] == allowed


def test_schema_mutation_preserves_and_reports():
    tf = make(celltype="int")
    assert tf.schema_celltypes["x"] == "int"
    with pytest.warns(CompiledPinCelltypeWarning):
        tf.schema = tf.schema.replace("dtype: int32", "dtype: float64")
    assert tf.celltypes.x == "int"
    assert "incompatible" in repr(tf)
    assert "incompatible" in repr(tf.schema_celltypes)
    with pytest.raises(TypeError):
        tf.schema_celltypes["x"] = "int"
    with pytest.raises(CompiledPinCelltypeError):
        tf(unknown_argument=object())  # Stage 1 precedes argument binding.
    with pytest.raises(CompiledPinCelltypeError):
        tf.celltypes.x = "plain"
    with pytest.raises(AttributeError):
        tf.celltypes.missing = "int"
    before = tf.schema
    with pytest.raises(yaml.YAMLError):
        tf.schema = "invalid: ["
    assert tf.schema == before


@pytest.mark.parametrize("celltype", ["mixed", "int", "binary"])
@pytest.mark.parametrize("raw", [b"null", b"null\n"])
def test_null_before_hash(celltype, raw):
    tf = make(celltype=celltype)
    with pytest.raises(CompiledPinSchemaError, match="'x'.*null"):
        tf(x=Buffer(raw).get_checksum())


@pytest.mark.parametrize("value", ["123", [1, 2], {"a": 1}, {"a": np.arange(3)}])
def test_mixed_taxonomy(value):
    with pytest.raises(CompiledMixedValueError):
        make()(x=value)


@pytest.mark.parametrize(
    "dtype,value,valid",
    [
        ("int32", np.int64(4), True),
        ("int32", np.float64(4), False),
        ("int32", np.int64(2**40), False),
        ("int32", True, False),
        ("float32", 2**24, True),
        ("float32", 2**24 + 1, False),
        ("float32", 1e300, False),
        ("float32", float("nan"), True),
        ("float32", float("inf"), True),
        ("float64", np.float32(1.5), True),
    ],
)
def test_scalar_admission(dtype, value, valid):
    tf = make(dtype)
    if valid:
        tf(x=value)
    else:
        with pytest.raises(CompiledPinSchemaError):
            tf(x=value)


@pytest.mark.parametrize("celltype", ["mixed", "binary"])
def test_zero_dim_by_kind(celltype):
    tf = make(celltype=celltype)
    p = tf._schema.inputs[0]
    for value, valid in [
        (np.int64(5), True),
        (np.float64(5), False),
        (np.array([5], dtype="int32"), False),
    ]:
        buf = Buffer(value, "binary")
        if valid:
            validate_pin(p, celltype, buf.get_checksum(), buffer=buf)
        else:
            with pytest.raises(CompiledPinSchemaError):
                validate_pin(p, celltype, buf.get_checksum(), buffer=buf)


@pytest.mark.parametrize(
    "celltype,value,count",
    [
        ("bytes", b"", 0),
        ("bytes", b"null\n", 0),
        ("bytes", b"\xe9\x00", 2),
        ("text", "", 0),
        ("text", "é", 2),
        ("binary", np.array([b"x", b"y"], dtype="S1"), 2),
    ],
)
def test_character_execution(celltype, value, count):
    tf = make("char", ["N"], celltype, direct=True)
    tf.code = "#include <stdint.h>\nint transform(unsigned int N, const unsigned char *x, int32_t *result) {*result=x ? N : -1;return 0;}"
    assert tf(x=value) == count


@pytest.mark.parametrize(
    "celltype,value", [("bytes", b"\xe9"), ("binary", np.array(b"\xe9", dtype="S1"))]
)
def test_unsigned_character_scalar(celltype, value):
    tf = make("char", celltype=celltype, direct=True)
    tf.code = "#include <stdint.h>\nint transform(unsigned char x, int32_t *result) {*result=x;return 0;}"
    assert tf(x=value) == 233


def test_noncanonical_null_bytes():
    tf = make("char", ["N"], "bytes")
    with pytest.raises(CompiledPinSchemaError, match="ambiguous checksum"):
        tf(x=Buffer(b"null").get_checksum())


def test_checksum_identity_and_no_input_fetch(monkeypatch):
    tf = make()
    cs = Checksum(sha256(b"remote integer buffer").hexdigest())
    original = Checksum.resolve

    def guarded(self, *a, **kw):
        assert self != cs, "input fetched while hashing"
        return original(self, *a, **kw)

    monkeypatch.setattr(Checksum, "resolve", guarded)
    a = tf(x=cs)
    first = a.construct()
    tf.celltypes.x = "int"
    second = tf(x=cs).construct()
    assert first != second


def test_conservative_pin_allows_liberal_typed_source():
    tf = make(celltype="int", direct=True)
    cell = Cell("mixed")
    cell.set(7)
    assert tf(x=cell) == 7


def test_pin_assignment_uses_serialized_value():
    tf = make("float64")
    tf.pins.x = np.float32(1.5)
    tf()
    with pytest.raises(CompiledPinSchemaError):
        make().pins.x = None


def test_replay_validates_before_compilation(monkeypatch):
    from seamless_transformer import compiler
    from seamless_transformer.run import call_compiled_transform

    tf = make()
    monkeypatch.setattr(
        compiler,
        "build_compiled_module",
        lambda *a: pytest.fail("compiled invalid input"),
    )
    d = {
        "__schema__": Buffer(tf.schema, "text").get_checksum().hex(),
        "__header__": Buffer(tf.header, "text").get_checksum().hex(),
        "__compilation__": Buffer({}, "plain").get_checksum().hex(),
    }
    with pytest.raises(CompiledPinSchemaError, match="missing input"):
        call_compiled_transform(d, {}, tf.code, {}, "mixed", {})
    nullbuf = Buffer(b" null \n")
    d["x"] = ("mixed", None, nullbuf.get_checksum().hex())
    with pytest.raises(CompiledPinSchemaError, match="null"):
        call_compiled_transform(d, {}, tf.code, {"x": None}, "mixed", {})


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reaches_native(dtype, value):
    tf = make(dtype, direct=True)
    tf.compilation.options.remove("-ffast-math")
    ctype = "float" if dtype == "float32" else "double"
    tf.code = f"#include <stdint.h>\n#include <math.h>\nint transform({ctype} x, int32_t *result) {{*result=isnan(x) ? 1 : isinf(x) ? 2 : 0;return 0;}}"
    assert tf(x=value) == (1 if np.isnan(value) else 2)


def test_complex_native():
    tf = make("complex64", direct=True)
    tf.code = "#include <stdint.h>\n#include <complex.h>\nint transform(float _Complex x, int32_t *result) {*result=crealf(x)+cimagf(x);return 0;}"
    assert tf(x=np.complex128(1 + 2j)) == 3


@pytest.mark.parametrize("value", [b"", b"ab", b"abcde"])
def test_fixed_character_length_rejected(value):
    tf = make("char", [4], "bytes")
    with pytest.raises(CompiledPinSchemaError):
        tf(x=value)


def test_character_content_and_text_edge_newline():
    tf = make("char", ["N"], "bytes", direct=True)
    tf.code = "#include <stdint.h>\nint transform(unsigned int N,const unsigned char *x,int32_t *result) {*result=N*1000+(N?x[0]:0);return 0;}"
    assert tf(x=b"\xe9\x00") == 2233
    source = Cell("text")
    source.set("ACGT")
    assert tf(x=source) == 5065
    tf.celltypes.x = "text"
    assert tf(x=source) == 4065


def test_shared_wildcard_literals_fail_before_hash():
    tf = make("int32", ["N"])
    tf.schema = tf.schema.replace(
        "outputs:", "- dtype: int32\n  name: y\n  shape: [N]\noutputs:"
    )
    with pytest.raises(CompiledPinSchemaError, match="wildcard"):
        tf(x=np.ones(2, dtype="int32"), y=np.ones(3, dtype="int32"))


def test_async_deferred_validation():
    import asyncio

    from seamless_transformer import delayed

    @delayed
    def upstream():
        return 9

    upstream.local = True
    tf = make(celltype="int")
    result = tf(x=upstream())
    asyncio.run(result.computation())
    assert result.run() == 9


@pytest.mark.parametrize(
    "dtype,shape",
    [
        ("int64", None),
        ("uint8", None),
        ("float64", None),
        ("bool", None),
        ("complex128", None),
        ("int16", ["N"]),
        ("char", None),
        ("char", ["N"]),
        ("char", ["N", 2]),
    ],
)
@pytest.mark.parametrize(
    "declaration", ["int", "float", "bool", "bytes", "text", "binary", "mixed"]
)
def test_every_whitelist_pair(dtype, shape, declaration):
    tf = make(dtype, shape, "binary")
    kind = parameter_kind(tf._schema.inputs[0])
    if declaration in DECLARATIONS[kind]:
        tf.celltypes.x = declaration
        tf._validate_compiled_stage1()
    else:
        with pytest.warns(CompiledPinCelltypeWarning):
            tf.celltypes.x = declaration
        with pytest.raises(CompiledPinCelltypeError):
            tf._validate_compiled_stage1()


def test_cache_hit_does_not_fetch_input(monkeypatch):
    tf = make()
    tf.code = "#include <stdint.h>\nint transform(int32_t x,int32_t *result) {*result=x+1;return 0;}"
    assert tf(x=517).run() == 518
    buf = Buffer(517, "mixed")
    cs = buf.get_checksum()
    original = Checksum.resolve

    def guarded(self, *args, **kwargs):
        assert self != cs, "cache hit fetched input data"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Checksum, "resolve", guarded)
    assert tf(x=cs).run() == 518


def test_pre_hash_and_executor_error_agree_without_hashtype():
    from seamless.checksum.hash_type import get_hash_type_cache

    tf = make()
    buf = Buffer(np.array([1], dtype="int32"), "binary")
    cs = buf.get_checksum()
    with pytest.raises(CompiledPinSchemaError) as early:
        validate_pin(tf._schema.inputs[0], "mixed", cs)
    get_hash_type_cache().pop(cs, None)
    with pytest.raises(CompiledPinSchemaError) as late:
        validate_pin(tf._schema.inputs[0], "mixed", cs, buffer=buf)
    assert str(early.value) == str(late.value)


def test_conservative_pin_dict_json_conversion_routes():
    import json

    from seamless import Expression

    tf = make("char", ["N"], "text", direct=True)
    tf.code = "#include <stdint.h>\nint transform(unsigned int N,const unsigned char *x,int32_t *result) {*result=N;return 0;}"
    value = {"answer": 42}
    source = Cell("plain")
    source.set(value)
    buf = Buffer(value, "plain")
    expected = len(buf.get_value("text").encode())
    assert tf(x=source) == expected
    assert (
        tf(x=Expression(buf.get_checksum(), input_celltype="plain", celltype="text"))
        == expected
    )
    text = json.dumps(value)
    assert tf(x=text) == len(text.encode())
    tf.pins.x.set_checksum(buf.get_checksum(), input_celltype="plain")
    assert tf() == expected


def test_conservative_pin_plain_list_and_projected_mixed_array():
    from seamless import Expression

    tf = make("int64", ["N"], "binary", direct=True)
    tf.code = "#include <stdint.h>\nint transform(unsigned int N,const int64_t *x,int32_t *result) {*result=N ? x[0] : 0;return 0;}"
    source = Cell("plain")
    source.set([21, 22])
    assert tf(x=source) == 21
    mixed = Cell("mixed")
    mixed.set({"array": np.array([31, 32], dtype="int64")})
    extracted = Expression(mixed, path=".array", celltype="binary")
    assert tf(x=extracted) == 31


def test_compiled_conversion_failure_names_pin_and_celltypes():
    from seamless_transformer.transformation_class import TransformationError

    tf = make("char", ["N"], "text", direct=True)
    source = Cell("bytes")
    source.set(b"\xff")
    with pytest.raises(
        (TransformationError, ValueError, TypeError), match="x.*bytes.*text"
    ):
        tf(x=source)
