"""Contract tests for compiled transformer input pins.

Oracle: ``seamless/docs/agent/contracts/compiled-pins.md``.  These tests fill
the gaps left by ``test_compiled_celltypes.py`` (which they do not repeat).
"""

import shutil
import warnings
from hashlib import sha256

import numpy as np
import pytest
import yaml

import seamless_transformer
from seamless import Buffer, Cell, Checksum
from seamless.checksum.conversion import SeamlessConversionError
from seamless_transformer import Transformer
from seamless_transformer.transformation_class import TransformationError
from seamless_transformer.compiled_validation import (
    CompiledMixedValueError,
    CompiledPinCelltypeError,
    CompiledPinCelltypeWarning,
    CompiledPinSchemaError,
)

pytestmark = pytest.mark.skipif(not shutil.which("gcc"), reason="gcc required")

INT_CODE = (
    "#include <stdint.h>\n"
    "int transform(int32_t x, int32_t *result) {*result=x; return 0;}"
)
BOOL_CODE = (
    "#include <stdint.h>\n#include <stdbool.h>\n"
    "int transform(bool x, int32_t *result) {*result=x?1:0; return 0;}"
)
F32_CODE = (
    "#include <stdint.h>\n"
    "int transform(float x, int32_t *result) {*result=(int)(x*10); return 0;}"
)
F64_CODE = (
    "#include <stdint.h>\n"
    "int transform(double x, int32_t *result) {*result=(int)(x*10); return 0;}"
)
C64_CODE = (
    "#include <stdint.h>\n#include <complex.h>\n"
    "int transform(float _Complex x, int32_t *result) {*result=crealf(x); return 0;}"
)
CHAR_SCALAR_CODE = (
    "#include <stdint.h>\n"
    "int transform(unsigned char x, int32_t *result) {*result=x; return 0;}"
)
CHAR_N_CODE = (
    "#include <stdint.h>\n"
    "int transform(unsigned int N, const unsigned char *x, int32_t *result)"
    " {*result=N; return 0;}"
)
CHAR_N2_CODE = (
    "#include <stdint.h>\n"
    "int transform(unsigned int N, const unsigned char *x, int32_t *result)"
    " {*result=N*100+x[1]; return 0;}"
)
# Length and position-weighted byte sum: pins the exact bytes, their order and
# their count.
CHAR_N_WEIGHTED_CODE = (
    "#include <stdint.h>\n"
    "int transform(unsigned int N, const unsigned char *x, int32_t *result) {\n"
    "  int32_t acc = 10000 * (int32_t)N;\n"
    "  for (unsigned int i = 0; i < N; i++) acc += (int32_t)(i + 1) * x[i];\n"
    "  *result = acc; return 0;\n"
    "}\n"
)


def char_n_weighted(data: bytes) -> int:
    """What CHAR_N_WEIGHTED_CODE returns when the kernel receives ``data``."""
    return 10000 * len(data) + sum((i + 1) * c for i, c in enumerate(data))



def make(dtype="int32", shape=None, celltype="mixed", *, code=INT_CODE, direct=True):
    """One-input compiled transformer; declaration warnings are not under test."""
    tf = Transformer("c", compiled=True, direct=direct)
    inp = {"name": "x", "dtype": dtype}
    if shape is not None:
        inp["shape"] = shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompiledPinCelltypeWarning)
        tf.schema = yaml.safe_dump(
            {"inputs": [inp], "outputs": [{"name": "result", "dtype": "int32"}]}
        )
        tf.celltypes.x = celltype
    tf.code = code
    return tf


def held_checksum(value, celltype=None):
    """Checksum whose buffer stays available to the local executor."""
    buf = Buffer(value, celltype) if celltype is not None else Buffer(value)
    buf.tempref()
    return buf.get_checksum()


def outcome(fn):
    """Run; return ("ok", result) or ("error", "<ClassName>: message").

    A pre-hash failure raises the compiled class directly; an executor failure
    arrives as a TransformationError whose text contains class name and
    message (§9, D5 rule (b)).  Both are normalized to text here.
    """
    try:
        return "ok", fn()
    except Exception as exc:  # noqa: BLE001
        return "error", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- §9 errors


def test_error_classes_are_public_typeerror_subclasses():
    """§9 D5, §12: public classes, subclassing TypeError; exported."""
    for name in (
        "CompiledPinCelltypeError",
        "CompiledMixedValueError",
        "CompiledPinSchemaError",
        "CompiledPinCelltypeWarning",
    ):
        assert hasattr(seamless_transformer, name), name
    for cls in (CompiledPinCelltypeError, CompiledMixedValueError, CompiledPinSchemaError):
        assert issubclass(cls, TypeError)
        assert getattr(seamless_transformer, cls.__name__) is cls
    assert issubclass(CompiledPinCelltypeWarning, Warning)


# ------------------------------------------------ rule 3: no optional pins


def test_optional_declaration_on_compiled_input_is_rejected():
    """Rule 3 / §9: an optional-pin declaration is rejected when declared."""
    tf = make(celltype="int")
    with pytest.raises((AttributeError, TypeError)):
        tf.optional_pins.x.enable()
    assert "x" not in tf.optional_pins


def test_optional_builder_state_is_rejected_before_binding():
    """Rule 3 / §5: an optional compiled input cannot reach Stage 1.

    No API route declares one and graph import rejects it, so this state is
    not reachable through the contract.  This is defence in depth: builder
    state injected directly must still be rejected before binding.
    """
    tf = make(celltype="int")
    tf._optional_pins.add("x")
    with pytest.raises(TypeError, match="optional"):
        tf(x=1)


# ---------------------------------------- §3c / §4 declarations & reports


@pytest.mark.parametrize(
    "celltype", ["plain", "str", "deepcell", "deepfolder", "folder", "checksum", "python", "yaml"]
)
def test_celltype_no_schema_allows_raises_when_declared(celltype):
    """§3c/§4: celltypes outside the seven raise immediately, state unchanged."""
    tf = make(celltype="int")
    with pytest.raises(CompiledPinCelltypeError):
        tf.celltypes.x = celltype
    assert tf.celltypes.x == "int"


def test_incompatible_diagnostic_names_every_required_item():
    """§4: pin, declared celltype, schema dtype and shape, schema celltype, allowed."""
    tf = make(celltype="int")
    with pytest.warns(CompiledPinCelltypeWarning) as record:
        tf.schema = tf.schema.replace("dtype: int32\n  name: x", "dtype: float64\n  name: x\n  shape:\n  - N")
    with pytest.raises(CompiledPinCelltypeError) as err:
        tf._validate_compiled_stage1()
    for message in (str(record[0].message), str(err.value)):
        assert "'x'" in message
        assert "'int'" in message
        assert "float64" in message
        assert "N" in message
        assert "'binary'" in message  # derived schema celltype
        assert "'mixed'" in message  # an allowed declaration


def test_missing_declaration_diagnostic_for_char_array():
    """§4: mixed on char[N] is reported as a missing declaration."""
    tf = make("int32", celltype="mixed")
    with pytest.warns(CompiledPinCelltypeWarning) as record:
        tf.schema = tf.schema.replace("dtype: int32\n  name: x", "dtype: char\n  name: x\n  shape:\n  - N")
    message = str(record[0].message)
    assert "'x'" in message and "explicit" in message
    for allowed in ("bytes", "text", "binary"):
        assert allowed in message
    assert tf.celltypes.x == "mixed"
    with pytest.raises(CompiledPinCelltypeError, match="explicit"):
        tf(x=b"ab")


def test_every_report_lists_all_incompatible_pins():
    """§4: every report lists all incompatible pins, not only the first."""
    tf = Transformer("c", compiled=True)
    tf.schema = (
        "inputs:\n  - {name: a, dtype: int32}\n  - {name: b, dtype: int32}\n"
        "outputs:\n  - {name: result, dtype: int32}\n"
    )
    tf.celltypes.a = "int"
    tf.celltypes.b = "int"
    with pytest.warns(CompiledPinCelltypeWarning) as record:
        tf.schema = tf.schema.replace("dtype: int32}", "dtype: float64, shape: [N]}", 2)
    reports = [str(w.message) for w in record if w.category is CompiledPinCelltypeWarning]
    assert any("'a'" in r and "'b'" in r for r in reports)
    with pytest.raises(CompiledPinCelltypeError) as err:
        tf(a=np.zeros(2), b=np.zeros(2))
    assert "'a'" in str(err.value) and "'b'" in str(err.value)
    assert "'a'" in repr(tf) and "'b'" in repr(tf)
    assert "'a'" in repr(tf.schema_celltypes) and "'b'" in repr(tf.schema_celltypes)


def test_mixed_declaration_stays_auto_across_schema_changes():
    """§4: never replace a mixed declaration; missing-declaration only on char."""
    tf = make("int32", celltype="mixed")
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompiledPinCelltypeWarning)
        tf.schema = tf.schema.replace("dtype: int32\n  name: x", "dtype: float64\n  name: x")
    assert tf.celltypes.x == "mixed"
    tf._validate_compiled_stage1()
    with pytest.warns(CompiledPinCelltypeWarning):
        tf.schema = tf.schema.replace("dtype: float64\n  name: x", "dtype: char\n  name: x\n  shape:\n  - N")
    assert tf.celltypes.x == "mixed"
    with pytest.raises(CompiledPinCelltypeError):
        tf._validate_compiled_stage1()
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompiledPinCelltypeWarning)
        tf.schema = tf.schema.replace("dtype: char\n  name: x\n  shape:\n  - N", "dtype: int32\n  name: x")
    assert tf.celltypes.x == "mixed"
    tf._validate_compiled_stage1()


def test_declaration_then_schema_either_order():
    """§4: change the declaration first, then the schema; no reset, no error."""
    tf = make("int32", celltype="int")
    with pytest.warns(CompiledPinCelltypeWarning):
        tf.celltypes.x = "float"
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompiledPinCelltypeWarning)
        tf.schema = tf.schema.replace("dtype: int32\n  name: x", "dtype: float64\n  name: x")
    assert tf.celltypes.x == "float"
    tf._validate_compiled_stage1()


def test_schema_that_fails_signature_validation_changes_nothing():
    """§4 step 1: a schema that does not parse (valid YAML, bad dtype) is atomic."""
    tf = make("int32", celltype="int")
    before_schema, before_header = tf.schema, tf.header
    with pytest.raises(Exception):
        tf.schema = (
            "inputs:\n  - {name: x, dtype: notadtype}\n"
            "outputs:\n  - {name: result, dtype: int32}\n"
        )
    assert tf.schema == before_schema
    assert tf.header == before_header
    assert tf.celltypes.x == "int"
    assert tf.schema_celltypes["x"] == "int"


def test_header_declares_schema_char_as_unsigned_char():
    """Rule 6: every schema char is the C type unsigned char."""
    tf = Transformer("c", compiled=True)
    with pytest.warns(CompiledPinCelltypeWarning):
        tf.schema = (
            "inputs:\n  - {name: s, dtype: char}\n  - {name: c, dtype: char, shape: [K]}\n"
            "  - {name: m, dtype: char, shape: [N, 2]}\n"
            "outputs:\n  - {name: result, dtype: int32}\n"
        )
    header = tf.header
    assert "unsigned char s" in header
    assert "const unsigned char *c" in header
    assert "typedef unsigned char char_2[2]" in header  # 2-D char element row
    import re

    # No plain C char anywhere: every 'char' token is 'unsigned char'.
    assert re.findall(r"(?<!unsigned )\bchar\b(?!_)", header) == [], header


# ------------------------------------------- §6 concrete transformation


def test_concrete_tuples_carry_declared_pin_celltypes():
    """§6: each input tuple carries its actual declared celltype; mixed stays mixed."""
    tf = Transformer("c", compiled=True)
    with pytest.warns(CompiledPinCelltypeWarning):
        tf.schema = (
            "inputs:\n  - {name: a, dtype: int32}\n  - {name: arr, dtype: float64, shape: [N]}\n"
            "  - {name: auto_pin, dtype: int32}\n  - {name: c, dtype: char, shape: [K]}\n"
            "  - {name: s, dtype: char}\n"
            "outputs:\n  - {name: result, dtype: int32}\n"
        )
    tf.celltypes.a = "int"
    tf.celltypes.arr = "binary"
    tf.celltypes.c = "text"
    tf.celltypes.s = "bytes"
    tf.code = "int transform(void) {return 0;}"
    transformation = tf(a=1, arr=np.zeros(2), auto_pin=3, c="hi", s=b"z")
    d = transformation.construct().resolve("plain")
    assert d["a"][0] == "int"
    assert d["arr"][0] == "binary"
    assert d["auto_pin"][0] == "mixed"
    assert d["c"][0] == "text"
    assert d["s"][0] == "bytes"
    assert "__schema__" in d


# ------------------------------------------------------- rule 4: null


@pytest.mark.parametrize(
    "dtype,celltype,code",
    [
        ("int32", "int", INT_CODE),
        ("float64", "float", F64_CODE),
        ("bool", "bool", BOOL_CODE),
        ("int32", "binary", INT_CODE),
        ("int32", "mixed", INT_CODE),
    ],
)
@pytest.mark.parametrize("raw", [b"null\n", b"null"])
def test_null_rejected_for_every_declared_scalar_celltype(dtype, celltype, code, raw):
    """Rule 4 / §9: null under any declared celltype is a targeted pin error."""
    tf = make(dtype, celltype=celltype, code=code)
    with pytest.raises(CompiledPinSchemaError) as err:
        tf(x=Buffer(raw).get_checksum())
    message = str(err.value)
    assert "'x'" in message and repr(celltype) in message and dtype in message
    assert "null" in message


@pytest.mark.parametrize("celltype", ["int", "float", "bool", "binary", "mixed"])
def test_python_none_literal_rejected(celltype):
    dtype = {"float": "float64", "bool": "bool"}.get(celltype, "int32")
    tf = make(dtype, celltype=celltype)
    with pytest.raises(CompiledPinSchemaError, match="'x'.*null"):
        tf(x=None)


def test_whitespace_padded_null_rejected_under_mixed():
    """Rule 4: detection is by reading; padded JSON null reads as None under mixed."""
    tf = make(celltype="mixed")
    kind, text = outcome(lambda: tf(x=held_checksum(b"  null \n")))
    assert kind == "error"
    assert "CompiledPinSchemaError" in text and "'x'" in text and "null" in text


def test_text_pin_rejects_string_null_and_passes_empty_string():
    """Rule 4 collision: text 'null' is the canonical null; '' has length 0."""
    tf = make("char", ["N"], "text", code=CHAR_N_CODE)
    with pytest.raises(CompiledPinSchemaError, match="'x'.*null"):
        tf(x="null")
    assert tf(x="") == 0


def test_length_zero_on_scalar_char_is_schema_error():
    """Rule 4: length 0 is a schema refinement error on a scalar char."""
    tf = make("char", None, "bytes", code=CHAR_SCALAR_CODE)
    with pytest.raises(CompiledPinSchemaError):
        tf(x=b"")
    with pytest.raises(CompiledPinSchemaError):
        tf(x=Buffer(b"null\n").get_checksum())  # canonical null reads as b"" under bytes


def test_noncanonical_null_on_bytes_names_both_origins_without_buffer(monkeypatch):
    """Rule 4: checksum-level ambiguity error naming both origins; no buffer fetch."""
    cs = Checksum(sha256(b"null").hexdigest())
    original = Checksum.resolve

    def guarded(self, *a, **kw):
        assert self != cs, "ambiguous-null check fetched the buffer"
        return original(self, *a, **kw)

    monkeypatch.setattr(Checksum, "resolve", guarded)
    tf = make("char", ["N"], "bytes", code=CHAR_N_CODE, direct=False)
    with pytest.raises(CompiledPinSchemaError) as err:
        tf(x=cs)
    message = str(err.value)
    assert "ambiguous" in message
    assert "4-byte" in message and "non-canonical" in message


# ------------------------------------------- §3a mixed admission table


@pytest.mark.parametrize(
    "dtype,shape,value,expected_hint",
    [
        ("float64", ["N"], [1.0, 2.0], "binary"),
        ("int32", None, "5", "int"),
        ("int32", None, {"a": 1}, "int"),
    ],
)
def test_mixed_rejection_lists_other_declarations(dtype, shape, value, expected_hint):
    """§9: CompiledMixedValueError lists the whitelist's other declarations."""
    tf = make(dtype, shape, "mixed")
    with pytest.raises(CompiledMixedValueError) as err:
        tf(x=value)
    assert "'x'" in str(err.value)
    assert expected_hint in str(err.value)


@pytest.mark.parametrize(
    "value,valid",
    [
        (True, True),
        (False, True),
        (np.bool_(True), True),
        (1, False),
        (np.int8(1), False),
        (1.0, False),
    ],
)
def test_bool_scalar_admission(value, valid):
    """§7: a Boolean parameter admits Python bool / bool dtype only."""
    tf = make("bool", celltype="mixed", code=BOOL_CODE)
    if valid:
        assert tf(x=value) == int(bool(value))
    else:
        with pytest.raises(CompiledPinSchemaError):
            tf(x=value)


def test_json_true_is_admitted_on_bool_despite_json_string_hashtype():
    """§5: a JSON_STRING HashType word does not prove a JSON string (true/false)."""
    tf = make("bool", celltype="mixed", code=BOOL_CODE)
    assert tf(x=Checksum(sha256(b"true\n").hexdigest())) == 1
    assert tf(x=Checksum(sha256(b"false\n").hexdigest())) == 0


@pytest.mark.parametrize(
    "raw,error",
    [
        (b"[]\n", CompiledMixedValueError),
        (b"{}\n", CompiledMixedValueError),
        (b'""\n', CompiledMixedValueError),
        (b"true\n", CompiledPinSchemaError),
    ],
)
def test_trivial_checksums_rejected_before_hash_without_fetch(monkeypatch, raw, error):
    """§5: checksum-level facts (trivial checksums, booleans) need no buffer."""
    cs = Checksum(sha256(raw).hexdigest())
    original = Checksum.resolve

    def guarded(self, *a, **kw):
        assert self != cs, "pre-hash check fetched a trivial buffer"
        return original(self, *a, **kw)

    monkeypatch.setattr(Checksum, "resolve", guarded)
    tf = make("int32", celltype="mixed", direct=False)
    with pytest.raises(error, match="'x'"):
        tf(x=cs)


# ---------------------------------------------- §3b auto consistency


def test_auto_consistency_int_schema_celltype():
    """§3b: an int-serialized value makes the same call on int and mixed pins."""
    cs = held_checksum(5, "int")
    assert make(celltype="int")(x=cs) == make(celltype="mixed")(x=cs) == 5


def test_int_reading_versus_mixed_on_noncanonical_float():
    """§3b/§7: int pin reads JSON 5.7 as 5; mixed pin rejects (no truncation)."""
    cs = held_checksum(b"5.7\n")
    assert make(celltype="int")(x=cs) == 5
    kind, text = outcome(lambda: make(celltype="mixed")(x=cs))
    assert kind == "error"
    assert "CompiledPinSchemaError" in text and "'x'" in text


# ---------------------------------------------------- §7 scalar rules


@pytest.mark.parametrize(
    "dtype,celltype,code,value,expected",
    [
        ("uint8", "mixed", INT_CODE.replace("int32_t x", "uint8_t x"), -1, None),
        ("uint8", "mixed", INT_CODE.replace("int32_t x", "uint8_t x"), 256, None),
        ("uint8", "mixed", INT_CODE.replace("int32_t x", "uint8_t x"), 255, 255),
        ("uint8", "binary", INT_CODE.replace("int32_t x", "uint8_t x"), np.int64(-1), None),
        ("float32", "mixed", F32_CODE, 0.1, 1),  # rounding within range accepted
        ("float32", "mixed", F32_CODE, 1e300, None),  # D2
        ("float32", "float", F32_CODE, 1e300, None),  # D2 under every celltype
        ("float32", "binary", F32_CODE, np.float64(1e300), None),  # D2
        ("float64", "mixed", F64_CODE, 2**53 + 1, None),  # D1
        ("float64", "mixed", F64_CODE, 2**50, "any"),  # D1 exact
        ("float32", "binary", F32_CODE, np.int64(2**24 + 1), None),  # D1 binary
        ("float32", "binary", F32_CODE, np.int16(7), 70),  # D3 by kind
        ("float64", "binary", F64_CODE, np.array(1.5, dtype=">f8"), 15),  # byte order ignored
        ("complex64", "mixed", C64_CODE, 1.0, None),  # no JSON complex
        ("complex64", "mixed", C64_CODE, np.complex128(1e300 + 0j), None),  # D2 per part
        ("complex64", "binary", C64_CODE, np.complex128(3 + 1j), 3),
    ],
)
def test_native_scalar_rules(dtype, celltype, code, value, expected):
    tf = make(dtype, celltype=celltype, code=code)
    if expected is None:
        with pytest.raises(CompiledPinSchemaError, match="'x'"):
            tf(x=value)
    elif expected == "any":
        tf(x=value)
    else:
        assert tf(x=value) == expected


def test_float_pin_cannot_hold_nonfinite():
    """§7/rule 5: int/float/bool celltypes cannot hold NaN or infinity."""
    tf = make("float32", celltype="float", code=F32_CODE)
    with pytest.raises((ValueError, TypeError)):
        tf(x=float("nan"))


def test_equivalent_scalars_same_native_call_different_checksums():
    """§7: 0-d float32/float64 .npy and JSON 1.5 differ in checksum, not in call."""
    tf = make("float64", celltype="mixed", code=F64_CODE)
    inputs = [
        held_checksum(np.float32(1.5), "binary"),
        held_checksum(np.float64(1.5), "binary"),
        held_checksum(1.5, "plain"),
    ]
    assert len({cs.hex() for cs in inputs}) == 3
    assert [tf(x=cs) for cs in inputs] == [15, 15, 15]


def test_literal_and_its_checksum_get_the_same_verdict():
    """§5: a literal and Buffer(value, pin_celltype).get_checksum() agree.

    The literal is rejected before hashing; the checksum may only be decidable
    in the executor.  Type and message must be the same either way.
    """
    tf = make(celltype="mixed")
    with pytest.raises(CompiledPinSchemaError) as early:
        tf(x=2**40)
    kind, text = outcome(lambda: tf(x=held_checksum(2**40, "mixed")))
    assert kind == "error"
    assert "CompiledPinSchemaError" in text
    assert str(early.value) in text


# --------------------------------------------------- §8 character rules


def test_scalar_char_requires_exactly_one_byte():
    tf = make("char", None, "bytes", code=CHAR_SCALAR_CODE)
    with pytest.raises(CompiledPinSchemaError):
        tf(x=b"ab")
    assert tf(x=b"A") == 65


def test_bytes_pin_encodes_str_literal_as_utf8():
    """§8 choosing table: str literals on a bytes pin are UTF-8."""
    assert make("char", ["N"], "bytes", code=CHAR_N_CODE)(x="é") == 2


def test_binary_char_pin_rejects_bytes_literal_and_json_string():
    """§8: bytes literal is unreadable as binary; a JSON string is rejected."""
    tf = make("char", ["N"], "binary", code=CHAR_N_CODE)
    with pytest.raises(CompiledPinSchemaError):
        tf(x=b"ab")
    kind, text = outcome(lambda: tf(x=held_checksum("ab", "plain")))
    assert kind == "error" and "'x'" in text


@pytest.mark.parametrize(
    "shape,celltype,value",
    [
        ([4], "binary", np.array(b"abcd", dtype="S4")),
        (["N", 4], "binary", np.array([b"abcd"], dtype="S4")),
        (["N", 4], "mixed", np.array([b"abcd"], dtype="S4")),
        (["N"], "binary", np.array([b"ab", b"cd"], dtype="S2")),
    ],
)
def test_binary_and_mixed_char_require_exact_s1(shape, celltype, value):
    """§8: S{k} with k > 1 is rejected even where its bytes would fit."""
    tf = make("char", shape, celltype, code=CHAR_N_CODE)
    with pytest.raises(CompiledPinSchemaError, match="'x'"):
        tf(x=value)


def test_bytes_cell_on_binary_char_array_is_rejected():
    """§8: bytes Cell → binary gives a 0-d S{len} array, rejected on char [N]."""
    source = Cell("bytes")
    source.set(b"ab")
    tf = make("char", ["N"], "binary", code=CHAR_N_CODE)
    kind, text = outcome(lambda: tf(x=source))
    assert kind == "error"
    assert "CompiledPinSchemaError" in text and "'x'" in text


def test_two_dimensional_char_array_accepts_mixed_s1():
    """§3c/§8: a char array with two or more dimensions keeps auto (mixed)."""
    tf = make("char", ["N", 2], "mixed", code=CHAR_N2_CODE)
    value = np.array([[b"a", b"b"], [b"c", b"d"]], dtype="S1")
    assert tf(x=value) == 2 * 100 + ord("b")


def test_bytes_pin_from_zero_dim_s_binary_cell():
    """§8: a 0-d S{len} binary Cell reaches a bytes pin as its raw bytes."""
    source = Cell("binary")
    source.set(np.array(b"abc", dtype="S3"))
    assert make("char", ["N"], "bytes", code=CHAR_N_CODE)(x=source) == 3


@pytest.mark.parametrize("cell_celltype", ["binary", "mixed"])
def test_bytes_pin_from_s1_array_cell_receives_tobytes(cell_celltype):
    """§8: a 1-D S1 binary/mixed Cell reaches a bytes pin as its raw bytes."""
    source = Cell(cell_celltype)
    source.set(np.array([b"a", b"b", b"c"], dtype="S1"))
    assert make("char", ["N"], "bytes", code=CHAR_N_CODE)(x=source) == 3


def test_text_pin_dict_literal_is_python_repr():
    """§2 worked example: a raw dict on a text pin gives the text checksum of str(d)."""
    d = {"answer": 42}
    tf = make("char", ["N"], "text", code=CHAR_N_CODE)
    assert tf(x=d) == len(str(d).encode())


# ------------------------------------------------ rule 2 / §4 inspection


def test_absent_schema_is_stage1_failure_and_schema_celltypes_empty():
    """Rule 2 / §4: no schema, no header, no transformer; schema_celltypes is
    empty while the schema is absent."""
    tf = Transformer("c", compiled=True)
    tf.code = INT_CODE
    assert dict(tf.schema_celltypes) == {}
    with pytest.raises(Exception, match="schema"):
        tf(x=1)


def test_schema_celltypes_is_read_only_mapping_independent_of_declarations():
    """§4 Inspecting: read-only mapping of every schema input to its schema
    celltype (None for char / 1-D char), derived from the schema alone; the
    transformer repr shows both celltypes."""
    from collections.abc import Mapping

    tf = Transformer("c", compiled=True)
    with pytest.warns(CompiledPinCelltypeWarning):
        tf.schema = (
            "inputs:\n  - {name: i, dtype: int32}\n  - {name: f, dtype: float32}\n"
            "  - {name: b, dtype: bool}\n  - {name: arr, dtype: float64, shape: [N]}\n"
            "  - {name: s, dtype: char}\n  - {name: c, dtype: char, shape: [K]}\n"
            "  - {name: m, dtype: char, shape: [K, 2]}\n"
            "outputs:\n  - {name: result, dtype: int32}\n"
        )
    expected = {
        "i": "int", "f": "float", "b": "bool", "arr": "binary",
        "s": None, "c": None, "m": "binary",
    }
    view = tf.schema_celltypes
    assert isinstance(view, Mapping)
    assert dict(view) == expected
    with pytest.raises(TypeError):
        view["i"] = "binary"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompiledPinCelltypeWarning)  # 's' stays undeclared
        tf.celltypes.i = "binary"
        tf.celltypes.c = "text"
    assert dict(tf.schema_celltypes) == expected  # declarations play no part
    text = repr(tf)
    assert "'binary'" in text and "'text'" in text  # declared
    assert "schema_celltypes" in text and "'int'" in text  # schema-derived


def test_explicit_mixed_declaration_is_the_default():
    """§3 "mixed is auto": declaring mixed is the same as the default and
    resets a pin to auto (same transformation checksum)."""
    default = make(celltype="mixed", direct=False)
    tf = Transformer("c", compiled=True, direct=False)
    tf.schema = default.schema
    tf.code = INT_CODE
    assert tf.celltypes.x == "mixed"  # new pins start as auto
    reset = make(celltype="int", direct=False)
    reset.celltypes.x = "mixed"
    checksums = {t(x=5).construct() for t in (default, tf, reset)}
    assert len(checksums) == 1


# ------------------------------------------ §3b auto consistency (all)


@pytest.mark.parametrize(
    "dtype,shape,schema_celltype,value,code,expected",
    [
        ("float64", None, "float", 1.5, F64_CODE, 15),
        ("bool", None, "bool", True, BOOL_CODE, 1),
        (
            "int32",
            ["N"],
            "binary",
            np.array([4, 5], dtype="int32"),
            "#include <stdint.h>\nint transform(unsigned int N, const int32_t *x,"
            " int32_t *result) {*result=N*10+x[1]; return 0;}",
            25,
        ),
    ],
)
def test_auto_consistency_every_schema_celltype(
    dtype, shape, schema_celltype, value, code, expected
):
    """§3b: a value serialized under the schema celltype makes the same call
    on a mixed pin as on a pin declared with the schema celltype."""
    cs = held_checksum(value, schema_celltype)
    on_schema = make(dtype, shape, schema_celltype, code=code)(x=cs)
    on_mixed = make(dtype, shape, "mixed", code=code)(x=cs)
    assert on_schema == on_mixed == expected


# ---------------------------------------- §8 binary/mixed -> bytes pins


@pytest.mark.parametrize("cell_celltype", ["binary", "mixed"])
@pytest.mark.parametrize(
    "value",
    [
        np.array([b"abcd", b"efgh"], dtype="S4"),
        np.array([[b"a", b"b"], [b"c", b"d"]], dtype="S1"),
        np.array([], dtype="S1"),
    ],
    ids=["S4-array", "2d-S1", "empty-S1"],
)
def test_bytes_pin_from_every_s_array_receives_tobytes(cell_celltype, value):
    """§8: conversion into a bytes pin calls tobytes() on every dtype-S array,
    whatever its shape and width; an empty one reads as b"" (length 0)."""
    source = Cell(cell_celltype)
    source.set(value)
    tf = make("char", ["N"], "bytes", code=CHAR_N_CODE)
    assert tf(x=source) == len(value.tobytes())


def test_bytes_pin_from_non_s_array_receives_whole_npy():
    """§8: a non-S array keeps its .npy checksum; the bytes pin receives the
    whole .npy buffer, header included."""
    value = np.arange(3, dtype="int32")
    source = Cell("binary")
    source.set(value)
    tf = make("char", ["N"], "bytes", code=CHAR_N_CODE)
    assert tf(x=source) == len(Buffer(value, "binary").content)


def test_bytes_pin_from_plain_cell_receives_json_text():
    """§8 choosing table: a plain Cell delivers its JSON text, quotes included."""
    source = Cell("plain")
    source.set("ab")
    tf = make("char", ["N"], "bytes", code=CHAR_N_CODE)
    assert tf(x=source) == len(Buffer("ab", "plain").content)  # '"ab"\n'


def test_text_pin_rejects_numpy_array_cell():
    """§8 choosing table: a text pin rejects NumPy arrays."""
    source = Cell("binary")
    source.set(np.array([b"a", b"b"], dtype="S1"))
    kind, text = outcome(lambda: make("char", ["N"], "text", code=CHAR_N_CODE)(x=source))
    assert kind == "error" and "x" in text


def test_text_cell_on_binary_char_array_is_rejected():
    """§8 choosing table: a text Cell on a binary char [N] pin is rejected.

    The contract attributes this to a 0-d S{len} array; per the conversion
    table text -> binary goes through mixed (a JSON string) and fails as a
    conversion error, so only the rejection is pinned here.
    """
    source = Cell("text")
    source.set("ab")
    kind, text = outcome(lambda: make("char", ["N"], "binary", code=CHAR_N_CODE)(x=source))
    assert kind == "error" and "x" in text


def test_text_cell_on_binary_char_array_pin_fails_with_conversion_error():
    """Unbound text Cell connected directly to a binary char [N] pin.

    text -> binary resolves through mixed (a JSON string) and raises
    SeamlessConversionError (author ruling: the converter is right).  The
    failure is the pin's own (pins.md, *Pin failures*): the pin is ``failed``,
    no transformation is built, so the kernel never runs.
    """
    text = Cell("text")
    text.set("ACGT")
    tf = make("char", ["N"], "binary", code=CHAR_N_WEIGHTED_CODE, direct=False)
    tf.pins.x = text
    pin = tf.pins.x
    assert pin.source is text
    assert pin.state == "failed"
    assert pin.checksum is None
    assert "cannot be converted from text to binary" in pin.exception
    with pytest.raises(SeamlessConversionError):
        pin.run()

    transformation = tf()
    assert transformation.construct() is None
    with pytest.raises(TransformationError, match="text.*binary"):
        transformation.run()


def test_text_cell_via_bytes_cell_on_bytes_char_array_runs():
    """Unbound text Cell -> unbound bytes Cell -> ``char [N]`` pin declared
    ``bytes``: the standard route for a text buffer (compiled-pins.md §8).

    ``text -> bytes`` is trivial, so the kernel receives the stored text buffer
    unchanged, trailing newline included: b"ACGT\\n", N = 5. The ``bytes``
    Cell is optional: the text Cell on the same pin delivers the same bytes.
    """
    text = Cell("text")
    text.set("ACGT")
    data = Cell("bytes", source=text)
    assert data.checksum == text.checksum
    assert data.value == b"ACGT\n"

    tf = make("char", ["N"], "bytes", code=CHAR_N_WEIGHTED_CODE, direct=False)
    tf.pins.x = data
    assert tf.pins.x.state == "complete"
    assert tf().run() == char_n_weighted(b"ACGT\n")  # 50798

    direct = make("char", ["N"], "bytes", code=CHAR_N_WEIGHTED_CODE, direct=False)
    direct.pins.x = text
    assert direct().run() == char_n_weighted(b"ACGT\n")


# ------------------------------------------- §5 placement, §6, §9 D5


def test_executor_failure_exception_string_carries_class_name():
    """§9 D5: a standalone transformation's .exception is a string that carries
    the class name for an executor-side compiled failure."""
    tf = make(celltype="mixed", direct=False)
    transformation = tf(x=held_checksum(2**40, "mixed"))
    transformation.compute()
    exc = transformation.exception
    assert isinstance(exc, str)
    assert "CompiledPinSchemaError" in exc and "'x'" in exc


def test_executor_rejection_is_never_cached():
    """§5 invariant: an invalid input never produces a cached result."""
    cs = held_checksum(2**40, "mixed")
    for _ in range(2):
        kind, text = outcome(lambda: make(celltype="mixed")(x=cs))
        assert kind == "error" and "CompiledPinSchemaError" in text


def test_async_deferred_rejection_matches_literal():
    """§6: synchronous and asynchronous (deferred) construction share the
    validation order; the rejection has the same class and message."""
    import asyncio

    from seamless_transformer import delayed

    @delayed
    def upstream():
        return [1, 2]

    upstream.local = True
    with pytest.raises(CompiledMixedValueError) as literal:
        make(celltype="mixed")(x=[1, 2])
    transformation = make(celltype="mixed", direct=False)(x=upstream())
    try:
        asyncio.run(transformation.computation())
    except Exception:  # noqa: BLE001
        pass
    exc = transformation.exception
    assert isinstance(exc, str)
    assert "CompiledMixedValueError" in exc and str(literal.value) in exc


def test_mixed_classification_is_reject_only():
    """§6: classification never rewrites a successful non-null value: the
    concrete tuple keeps the input's mixed checksum."""
    cs = held_checksum(3, "mixed")
    tf = make(celltype="mixed", direct=False)
    d = tf(x=cs).construct().resolve("plain")
    assert d["x"] == ("mixed", None, cs.hex()) or list(d["x"]) == ["mixed", None, cs.hex()]
