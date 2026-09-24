"""Contract coverage for deep-celltypes.md, pin layer and output side.

§How deep values reach a transformer and §The output side: a deep result is an index.
Known gaps are non-strict xfails that assert the contract, never the bug.
"""
import pytest

from seamless import Buffer, Checksum
from seamless.transformer import delayed, direct
from seamless_transformer import transformation_utils

DOC = "deep-celltypes.md"

PIN_GAP = pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §How deep values reach a transformer: a deep input pin fails in "
    "pretransformation._to_checksum, which asks HashType validate_deserializable_as "
    "with the deep name and gets 'celltype is outside the HashType domain'",
)

RESULT_TYPING_GAP = pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §What a deep buffer is / §The output side: the value of a deep "
    "result is {key: Checksum}; the code hands back raw hex strings",
)


def _held(value, celltype=None):
    buffer = Buffer(value, celltype) if celltype else Buffer(value)
    buffer.tempref()
    return buffer


def _describe_pin():
    @direct
    def describe(x):
        return {
            key: [type(value).__name__, value.hex() if hasattr(value, "hex") and not isinstance(value, bytes) else value.decode()]
            for key, value in x.items()
        }

    return describe


# --- Input side ------------------------------------------------------------


@PIN_GAP
@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder"])
def test_deepcell_and_deepfolder_pins_hand_over_unresolved_checksums(celltype):
    member_content = b"deep pin member, never stored"
    from seamless.checksum.calculate_checksum import calculate_checksum

    member = Checksum(calculate_checksum(member_content))
    index = _held({"dir/k": member.hex()}, "plain")
    describe = _describe_pin()
    describe.celltypes.x = celltype
    # The member buffer does not exist: success proves no resolution.
    assert describe(x=index.get_checksum()) == {"dir/k": ["Checksum", member.hex()]}


@PIN_GAP
def test_folder_pin_hands_over_resolved_child_contents():
    child = _held(b"folder child contents")
    index = _held({"dir/k": child.get_checksum().hex()}, "plain")
    describe = _describe_pin()
    describe.celltypes.x = "folder"
    assert describe(x=index.get_checksum()) == {"dir/k": ["bytes", "folder child contents"]}


@PIN_GAP
@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder", "folder"])
def test_deep_pin_rejects_a_nested_index_through_the_shared_validator(celltype):
    child = _held(b"nested child")
    nested = _held({"outer": {"inner": child.get_checksum().hex()}}, "plain")
    describe = _describe_pin()
    describe.celltypes.x = celltype
    with pytest.raises(Exception, match="nested"):
        describe(x=nested.get_checksum())


@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder", "folder"])
@pytest.mark.parametrize(
    "bad,match",
    [({"k": "AA" * 32}, "k"), ({"k": "short"}, "k"), ({1: "aa" * 32}, "1")],
    ids=["uppercase", "short", "int-key"],
)
def test_pin_unpacking_uses_the_same_validator_failures_as_expressions(celltype, bad, match):
    with pytest.raises(ValueError, match=match):
        transformation_utils.unpack_deep_structure(bad, celltype)


@pytest.mark.xfail(
    strict=False,
    reason=f"{DOC} §Nesting is not contract: the deepcell/deepfolder pin presentation "
    "(transformation_namespace._to_checksum_dict) still recurses through nesting instead "
    "of calling the shared validator",
)
def test_pin_namespace_checksum_dict_rejects_nesting_like_unpacking():
    """One shared validator: the deepcell/deepfolder presentation must refuse what unpacking refuses."""
    from seamless_transformer.transformation_namespace import _to_checksum_dict

    nested = {"outer": {"inner": "aa" * 32}}
    with pytest.raises(ValueError):
        transformation_utils.unpack_deep_structure(nested, "deepcell")
    with pytest.raises(ValueError, match="nested"):
        _to_checksum_dict(nested)


# --- Output side -----------------------------------------------------------


def test_deepcell_result_checksum_is_the_index_checksum():
    @delayed
    def produce():
        return {"a": 5, "b": "text"}

    produce.celltypes.result = "deepcell"
    transformation = produce()
    result_checksum = transformation.compute()
    expected = Buffer(
        {"a": Buffer(5, "mixed").get_checksum().hex(), "b": Buffer("text", "mixed").get_checksum().hex()},
        "deepcell",
    ).get_checksum()
    assert result_checksum == expected


def test_folder_result_is_an_index_of_the_produced_bytes():
    @delayed
    def produce():
        return {"x.txt": b"hello", "y/z.bin": b"\x00\x01"}

    produce.celltypes.result = "folder"
    transformation = produce()
    index = transformation.run()
    assert {key: Checksum(value).hex() for key, value in index.items()} == {
        "x.txt": Buffer(b"hello").get_checksum().hex(),
        "y/z.bin": Buffer(b"\x00\x01").get_checksum().hex(),
    }


@RESULT_TYPING_GAP
@pytest.mark.parametrize("celltype", ["deepcell", "folder"])
def test_delayed_deep_result_value_is_an_index_of_checksum_objects(celltype):
    @delayed
    def produce():
        return {"a": b"one"}

    produce.celltypes.result = celltype
    value = produce().run()
    assert isinstance(value["a"], Checksum)


@RESULT_TYPING_GAP
@pytest.mark.parametrize("celltype", ["deepcell", "folder"])
def test_direct_deep_result_is_an_unresolved_index_of_checksum_objects(celltype):
    @direct
    def produce():
        return {"a": b"one", "b": b"two"}

    produce.celltypes.result = celltype
    value = produce()
    assert set(value) == {"a", "b"}
    assert all(isinstance(member, Checksum) for member in value.values())


def test_direct_deep_result_resolves_no_children():
    """No fan-out on the way out: members are references, not materialized values."""

    @direct
    def produce():
        return {"a": "one", "b": "two"}

    produce.celltypes.result = "deepcell"
    value = produce()
    member_hexes = {Checksum(member).hex() for member in value.values()}
    assert member_hexes == {
        Buffer("one", "mixed").get_checksum().hex(),
        Buffer("two", "mixed").get_checksum().hex(),
    }
