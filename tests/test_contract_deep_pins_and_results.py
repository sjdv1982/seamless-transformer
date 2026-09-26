"""Contract coverage for deep-celltypes.md, pin layer and output side.

§How deep values reach a transformer and §The output side: a deep result is an index.
"""
import pytest

from seamless import Buffer, Checksum
from seamless.transformer import delayed, direct
from seamless_transformer import transformation_utils

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


def test_folder_pin_hands_over_resolved_child_contents():
    child = _held(b"folder child contents")
    index = _held({"dir/k": child.get_checksum().hex()}, "plain")
    describe = _describe_pin()
    describe.celltypes.x = "folder"
    assert describe(x=index.get_checksum()) == {"dir/k": ["bytes", "folder child contents"]}


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


def test_pin_namespace_checksum_dict_rejects_nesting_like_unpacking():
    """One shared validator: the deepcell/deepfolder presentation must refuse what unpacking refuses."""
    from seamless_transformer.transformation_namespace import _to_checksum_dict

    nested = {"outer": {"inner": "aa" * 32}}
    with pytest.raises(ValueError):
        transformation_utils.unpack_deep_structure(nested, "deepcell")
    with pytest.raises(ValueError, match="nested"):
        _to_checksum_dict(nested)


@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder", "folder"])
def test_only_deepfolder_and_folder_pins_carry_the_directory_filesystem_format(celltype):
    """'deepfolder and folder pins additionally carry {"filesystem": {"mode": "directory"}}'."""

    @delayed
    def consume(d):
        return 1

    consume.celltypes.d = celltype
    index = _held({"k": "aa" * 32}, "plain")
    transformation_dict = consume(index.get_checksum()).construct().resolve("plain")
    fmt = transformation_dict.get("__format__", {}).get("d")
    if celltype == "deepcell":
        assert fmt is None
    else:
        assert fmt == {"celltype": celltype, "filesystem": {"mode": "directory"}}


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
    assert all(type(member) is Checksum for member in index.values())
    assert index == {
        "x.txt": Buffer(b"hello").get_checksum(),
        "y/z.bin": Buffer(b"\x00\x01").get_checksum(),
    }


@pytest.mark.parametrize("celltype", ["deepcell", "folder"])
def test_delayed_deep_result_value_is_an_index_of_checksum_objects(celltype):
    @delayed
    def produce():
        return {"a": b"one"}

    produce.celltypes.result = celltype
    value = produce().run()
    assert isinstance(value["a"], Checksum)


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
    assert all(type(member) is Checksum for member in value.values())
    assert value == {
        "a": Buffer("one", "mixed").get_checksum(),
        "b": Buffer("two", "mixed").get_checksum(),
    }


@pytest.mark.parametrize("mode", ["direct", "delayed"])
def test_deepcell_result_members_may_be_dicts_and_lists(mode):
    """Ruling 2026-09-26: refusing dict/list member values of a deepcell result is a bug.

    A member is a mixed value; the index stays flat (key -> member checksum).
    """
    decorator = direct if mode == "direct" else delayed

    @decorator
    def produce():
        return {"a": {"v": 2}, "b": [1, 2]}

    produce.celltypes.result = "deepcell"
    value = produce() if mode == "direct" else produce().run()
    assert all(type(member) is Checksum for member in value.values())
    assert value == {
        "a": Buffer({"v": 2}, "mixed").get_checksum(),
        "b": Buffer([1, 2], "mixed").get_checksum(),
    }


@pytest.mark.parametrize("celltype", ["deepfolder", "module"])
def test_result_may_not_be_declared_deepfolder_or_module(celltype):
    @delayed
    def produce():
        return {}

    with pytest.raises(TypeError):
        produce.celltypes.result = celltype


@pytest.mark.parametrize("celltype", ["deepcell", "folder"])
def test_result_may_be_declared_deepcell_or_folder(celltype):
    @delayed
    def produce():
        return {}

    produce.celltypes.result = celltype
    assert produce.celltypes.result == celltype
