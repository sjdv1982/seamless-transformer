"""Standalone Pin contract coverage (seamless/docs/agent/contracts/pins.md).

Fills the gaps left by test_pin_handles.py, test_pin_conversion.py,
test_optional_pin_contract.py and test_transformation_checksum.py: the rules
stated for "any"/"every" celltype are walked over the celltype list, and the
pin-path rules are checked on the standalone Transformer too.
"""
import pytest
from seamless import AuthorityError, Buffer, CacheMissError, Cell, Checksum
from seamless_transformer import delayed
from seamless_transformer.transformation_class import TransformationError

DOC = "pins.md"
NULL = Buffer(None, "plain").get_checksum()

NULLABLE = ["plain", "mixed", "bytes"]
# yaml / python are accepted by CelltypesWrapper but raise NotImplementedError
# when a transformation is built (pretransformation.py); they are exercised
# separately below.
NON_NULLABLE_INPUT = ["binary", "int", "float", "bool", "str", "text", "ipython",
                      "checksum", "deepcell", "deepfolder", "folder", "module"]
NON_NULLABLE_RESULT = ["binary", "int", "float", "bool", "str", "text", "checksum",
                       "deepcell", "folder"]


def identity(value):
    return value


def optional_identity(value=None):
    return value


def builder(func=identity, celltype=None):
    tf = delayed(func)
    tf.local = True
    if celltype is not None:
        tf.celltypes.value = celltype
    return tf


# --- Reaching pins ---------------------------------------------------------

def test_args_is_an_exact_alias_of_pins():
    tf = builder(celltype="int")
    tf.args.value = 4
    assert tf.pins.value.value == 4
    tf.pins.value = 5
    assert tf.args.value.value == 5
    assert dir(tf.args) == dir(tf.pins)
    tf.args.value.checksum = None
    assert tf.pins.value.state == "unwired"


def test_standalone_transformer_has_no_pin_sugar():
    tf = builder()
    with pytest.raises(AttributeError, match=r"reached as \.pins\['value'\]"):
        tf.value = 1
    with pytest.raises(AttributeError):
        tf.value
    with pytest.raises(TypeError):
        tf["value"]
    with pytest.raises(TypeError):
        tf["value"] = 1
    assert not hasattr(type(tf), "__getitem__")
    assert tf.pins.value.state == "unwired"


def test_item_form_is_not_an_escape_hatch():
    tf = builder()
    with pytest.raises(AttributeError):
        tf.pins["typo"]
    with pytest.raises(AttributeError):
        tf.pins["typo"] = 1
    with pytest.raises(AttributeError):
        tf.pins.typo = 1
    assert "typo" not in dir(tf.pins)


@pytest.mark.parametrize("code", ["callable", "signatureless"])
def test_result_is_never_a_pin_name(code):
    tf = builder() if code == "callable" else delayed("result = value")
    operations = [
        lambda: tf.pins.result,
        lambda: tf.pins["result"],
        lambda: setattr(tf.pins, "result", 1),
        lambda: tf.pins.__setitem__("result", 1),
        lambda: tf.pins.__delitem__("result"),
        lambda: delattr(tf.pins, "result"),
    ]
    for operation in operations:
        with pytest.raises(AttributeError):
            operation()
    assert "result" not in dir(tf.pins)
    # `result` is the output celltype slot.
    tf.celltypes.result = "int"
    assert tf.celltypes.result == "int"


# --- Which pins exist ------------------------------------------------------

def test_signatureless_celltype_declares_and_del_celltype_removes():
    tf = delayed("result = x")
    tf.celltypes.x = "int"
    assert "x" in dir(tf.pins)
    assert tf.pins.x.state == "unwired"
    assert tf.pins.x.celltype == "int"
    tf.pins.x = 3
    assert tf.pins.x.state == "complete"
    del tf.celltypes.x
    assert "x" not in dir(tf.pins)
    assert "x" not in tf.optional_pins
    with pytest.raises(AttributeError):
        tf.pins.x
    # Re-declaring does not resurrect the old input.
    tf.celltypes.x = "int"
    assert tf.pins.x.state == "unwired"


def test_signature_fixes_the_pin_set_for_celltypes_too():
    tf = builder()
    with pytest.raises(AttributeError):
        tf.celltypes.typo = "int"
    with pytest.raises(AttributeError):
        del tf.celltypes.value


# --- Celltypes -------------------------------------------------------------

@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder", "folder", "module"])
def test_input_pin_slot_accepts_deep_and_module_celltypes(celltype):
    tf = builder()
    tf.celltypes.value = celltype
    assert tf.pins.value.celltype == celltype
    tf2 = builder()
    tf2.pins.value.celltype = celltype
    assert tf2.celltypes.value == celltype


def test_celltype_setter_accepts_python_type_on_both_sides():
    tf = builder()
    tf.celltypes.value = float
    assert tf.pins.value.celltype == "float"
    tf.pins.value.celltype = str
    assert tf.celltypes.value == "str"


# --- Pins hold checksums ---------------------------------------------------

def test_bare_checksum_input_celltype_defaults_to_pin_celltype():
    buffer = Buffer(42, "int")
    buffer.tempref()
    checksum = buffer.get_checksum()
    tf = builder(celltype="str")
    tf.pins.value.set_checksum(checksum)
    pin = tf.pins.value
    assert pin.input_celltype == "str"
    assert pin.source is None
    tf2 = builder(celltype="str")
    tf2.pins.value = checksum
    assert tf2.pins.value.input_celltype == "str"
    assert tf2.pins.value.checksum == checksum


def test_checksum_is_a_value_only_for_celltype_checksum():
    buffer = Buffer(5, "int")
    buffer.tempref()
    checksum = buffer.get_checksum()
    tf = builder(celltype="checksum")
    tf.celltypes.result = "checksum"
    tf.pins.value = checksum
    pin = tf.pins.value
    # A value: serialized as the checksum's hex string, not a reference to it.
    assert pin.value == checksum.hex()
    assert pin.checksum != checksum
    assert tf().run() == checksum.hex()
    reference = builder(celltype="int")
    reference.pins.value = checksum
    assert reference.pins.value.checksum == checksum
    assert reference.pins.value.value == 5


def test_conversion_is_skipped_when_celltypes_are_equal(monkeypatch):
    from seamless_transformer import pin_class

    class NoExpression:
        def __init__(self, *args, **kwargs):
            raise AssertionError("an equal-celltype pin must not build an Expression")

    tf = builder(celltype="int")
    tf.pins.value = 7
    monkeypatch.setattr(pin_class, "Expression", NoExpression)
    assert tf.pins.value.checksum == Buffer(7, "int").get_checksum()
    tf.pins.value.celltype = "str"
    assert tf.pins.value.checksum is None
    assert "must not build an Expression" in tf.pins.value.exception


# --- Writes ----------------------------------------------------------------

def test_set_checksum_none_clears_and_keeps_declaration():
    tf = builder(celltype="int")
    tf.pins.value = 3
    tf.pins.value.set_checksum(None)
    assert tf.pins.value.state == "unwired"
    assert tf.pins.value.input_celltype is None
    assert tf.celltypes.value == "int"


def test_authority_error_message():
    upstream = Cell("int")
    upstream.set(1)
    tf = builder(celltype="int")
    tf.pins.value = upstream
    expected = ("The pin is controlled by a source; assign .value, .buffer or "
                ".checksum to replace it")
    for method, value in [("set", 2), ("set_buffer", Buffer(2, "int")),
                          ("set_checksum", Buffer(2, "int").get_checksum())]:
        with pytest.raises(AuthorityError) as excinfo:
            getattr(tf.pins.value, method)(value)
        assert str(excinfo.value) == expected


# --- Null, required pins, optional pins ------------------------------------

@pytest.mark.parametrize("celltype", NULLABLE)
def test_required_nullable_pin_accepts_null(celltype):
    tf = builder(celltype=celltype)
    tf.pins.value = None
    assert tf.pins.value.checksum == NULL
    tf2 = builder(celltype=celltype)
    tf2.pins.value.set(None)
    assert tf2.pins.value.checksum == NULL
    expected = b"" if celltype == "bytes" else None
    assert tf().run() == expected


@pytest.mark.parametrize("celltype", NON_NULLABLE_INPUT)
def test_required_pin_rejects_null_for_every_other_celltype(celltype):
    message = f"Required pin 'value' with celltype '{celltype}' cannot accept null"
    tf = builder(celltype=celltype)
    with pytest.raises(TypeError) as excinfo:
        tf.pins.value = None
    assert str(excinfo.value) == message
    with pytest.raises(TypeError, match="cannot accept null"):
        tf.pins.value.set(None)
    assert tf.pins.value.state == "unwired"


_MODULE_NULL = pytest.mark.xfail(strict=False, reason=(
    "pins.md §Null, required pins (null from upstream is reported on the pin) and "
    "§Conversion (no transformation is built): a module pin is serialized as "
    "celltype 'plain' in the transformation dict, so validate_pin_null lets the "
    "null through and the transformation constructs although the pin is failed"))


@pytest.mark.parametrize("celltype", [
    pytest.param(ct, marks=_MODULE_NULL) if ct == "module" else ct
    for ct in NON_NULLABLE_INPUT])
def test_required_pin_null_from_upstream_is_reported_on_the_pin(celltype):
    upstream = Cell("plain")
    upstream.set(None)
    tf = builder(celltype=celltype)
    tf.pins.value = upstream
    pin = tf.pins.value
    assert pin.checksum is None
    assert pin.state == "failed"
    assert f"Required pin 'value' with celltype '{celltype}' cannot accept null" in pin.exception
    transformation = tf()
    assert transformation.construct() is None
    assert "Required pin 'value'" in transformation.exception


@pytest.mark.parametrize("celltype", NON_NULLABLE_RESULT)
def test_null_result_rejected_for_every_other_celltype(celltype):
    @delayed
    def returns_none():
        return None

    returns_none.local = True
    returns_none.celltypes.result = celltype
    with pytest.raises(TransformationError,
                       match=f"Null result is not allowed for celltype '{celltype}'"):
        returns_none().run()


def _null_upstream():
    @delayed
    def returns_none():
        return None

    returns_none.local = True
    returns_none.celltypes.result = "plain"
    return returns_none()


def _null_cell():
    cell = Cell("plain")
    cell.set(None)
    return cell


ROUTES = {
    "null-checksum": lambda: NULL,
    "null-cell": _null_cell,
    "null-transformation": _null_upstream,
}

_FORMAT = pytest.mark.xfail(strict=False, reason=(
    "pins.md §Null, required pins, optional pins (identity rule): the dropped "
    "folder/deepfolder pin leaves its __format__ entry in the transformation dict"))
_HASHTYPE = pytest.mark.xfail(strict=False, reason=(
    "pins.md §Null (identity rule, drop before conversion): a bare null Checksum on "
    "a deep optional pin raises 'celltype is outside the HashType domain'"))
_ILLEGAL = pytest.mark.xfail(strict=False, reason=(
    "pins.md §Null (drop happens before conversion): a null Transformation argument "
    "is wrapped in an illegal plain->deep/module conversion before the drop"))


def _identity_cases():
    cases = []
    for celltype in NULLABLE + NON_NULLABLE_INPUT:
        for route in ROUTES:
            marks = []
            if celltype in ("folder", "deepfolder"):
                marks.append(_FORMAT)
            if route == "null-checksum" and celltype in ("deepcell", "deepfolder", "folder"):
                marks.append(_HASHTYPE)
            if route == "null-transformation" and celltype in (
                    "deepcell", "deepfolder", "folder", "module"):
                marks.append(_ILLEGAL)
            if route == "null-checksum" and celltype == "checksum":
                continue  # a Checksum is a value there; see the test below
            cases.append(pytest.param(celltype, route, marks=marks,
                                      id=f"{celltype}-{route}"))
    return cases


@pytest.mark.parametrize("celltype,route", _identity_cases())
def test_optional_null_has_absent_identity_for_every_celltype(celltype, route):
    tf = builder(optional_identity, celltype)
    assert tf.optional_pins == {"value"}
    absent = tf().construct()
    assert absent is not None
    connected = tf(ROUTES[route]())
    assert connected.construct() == absent, connected.exception
    assert connected.run() is None


@pytest.mark.parametrize("celltype", NULLABLE + NON_NULLABLE_INPUT)
def test_prebound_optional_null_has_absent_identity(celltype):
    if celltype in ("folder", "deepfolder"):
        pytest.xfail(_FORMAT.kwargs["reason"])
    tf = builder(optional_identity, celltype)
    absent = tf().construct()
    tf.pins.value = _null_cell()
    assert tf().construct() == absent
    tf2 = builder(optional_identity, celltype)
    tf2.pins.value = None
    assert tf2.pins.value.input_celltype == celltype
    assert tf2().construct() == absent


def test_null_checksum_on_optional_checksum_pin_is_a_value():
    tf = builder(optional_identity, "checksum")
    tf.celltypes.result = "checksum"
    absent = tf().construct()
    present = tf(NULL)
    assert present.construct() != absent
    assert present.run() == NULL.hex()


@pytest.mark.parametrize("celltype", NULLABLE)
def test_required_null_and_optional_null_are_different_identities(celltype):
    tf = builder(optional_identity, celltype)
    absent = tf().construct()
    assert tf(None).construct() == absent
    tf.optional_pins.value.disable()
    required_null = tf(None)
    assert required_null.construct() != absent
    assert required_null.run() == (b"" if celltype == "bytes" else None)


@pytest.mark.parametrize("celltype", ["int", "text", "binary"])
def test_non_null_optional_pin_participates_in_identity(celltype):
    values = {"int": 5, "text": "five", "binary": __import__("numpy").arange(3)}
    tf = builder(optional_identity, celltype)
    assert tf(values[celltype]).construct() != tf().construct()


def test_optionality_is_not_in_the_hashed_payload():
    tf = builder(optional_identity, "int")
    enabled = tf(5).construct()
    tf.optional_pins.value.disable()
    assert tf(5).construct() == enabled
    tf.optional_pins.value.enable()
    assert tf(5).construct() == enabled


def test_optional_pins_item_access_for_names_colliding_with_methods():
    def colliding(add=1, disable=2, enable=3, x=4):
        return [add, disable, enable, x]

    tf = builder(colliding)
    view = tf.optional_pins
    assert view == {"add", "disable", "enable", "x"}
    view["add"].disable()
    view["disable"].disable()
    assert view == {"enable", "x"}
    assert "add" in dir(view) and "disable" in dir(view)
    view["add"].enable()
    view["disable"].enable()
    assert view == {"add", "disable", "enable", "x"}
    assert tf().run() == [1, 2, 3, 4]


def test_optional_pin_default_is_received_not_prebound():
    def default_seven(value=7):
        return value

    tf = builder(default_seven, "int")
    assert tf.pins.value.state == "unwired"
    assert tf().run() == 7
    assert "value" not in tf().construct().resolve("plain")


# --- Reads -----------------------------------------------------------------

def test_cachemiss_on_materialization_is_not_recorded():
    tf = builder(celltype="int")
    tf.pins.value.set_checksum(Checksum("ab" * 32))
    pin = tf.pins.value
    with pytest.raises(CacheMissError):
        pin.value
    assert pin.exception is None
    with pytest.raises(CacheMissError):
        pin.buffer
    assert tf.pins.value.exception is None


def test_pin_surface_has_no_cell_only_members():
    pin = builder(celltype="int").pins.value
    for name in ("path", "prune", "mount", "validator"):
        assert not hasattr(pin, name), name
    for name in ("source", "checksum", "buffer", "value", "celltype", "input_celltype",
                 "state", "exception", "clear_exception", "build", "compute",
                 "compute_async", "run", "fingertip", "set", "set_buffer", "set_checksum"):
        assert hasattr(type(pin), name), name


def test_bytes_pin_run_returns_raw_bytes():
    tf = builder(celltype="bytes")
    tf.pins.value = b"abc"
    assert tf.pins.value.run() == b"abc"


def test_fingertip_on_unwired_pin_is_a_noop():
    pin = builder(celltype="int").pins.value
    assert pin.fingertip() is None
    assert pin.exception is None
