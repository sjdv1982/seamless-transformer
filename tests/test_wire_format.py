"""Host-side wire format contract tests (Phase 3b)."""

import builtins
import io
import json

import numpy as np
import pytest

from seamless import Buffer, Checksum
from seamless_transformer.wire_format import (
    decode_jsonarray,
    read_wire_directory,
    write_checksum_pin_index,
    write_deepcell_pin_index,
)


def _index(index, **files):
    return {"index.json": json.dumps(index).encode(), **files}


def _npy(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def test_read_value_and_members_all_encodings():
    checksum = Buffer("member", "mixed").get_checksum()
    mixed = Buffer({"x": [1, 2]}, "mixed").content
    array = np.array([[1, 2], [3, 4]], dtype=np.int64)
    files = _index(
        {
            "members": {
                "checksum": {"__checksum__": checksum.hex()},
                "json": "m0.json",
                "bytes": "m1.bytes",
                "npy": "m2.npy",
                "jsonarray": "m3.jsonarray",
                "mixed": "m4.mixed",
            }
        },
        **{
            "m0.json": b'{"a":1}',
            "m1.bytes": b"\x00\xff",
            "m2.npy": _npy(array),
            "m3.jsonarray": json.dumps(
                {"dtype": "int64", "shape": [2], "values": [5, 6]}
            ).encode(),
            "m4.mixed": mixed,
        },
    )

    result = read_wire_directory(files, "deepcell")
    assert result["checksum"] == checksum
    assert result["json"] == {"a": 1}
    assert result["bytes"] == b"\x00\xff"
    np.testing.assert_array_equal(result["npy"], array)
    np.testing.assert_array_equal(result["jsonarray"], np.array([5, 6]))
    assert result["mixed"] == {"x": [1, 2]}
    assert read_wire_directory(_index({"value": "v.json"}, **{"v.json": b"42"}), "mixed") == 42


def test_checksum_entries_decode_to_checksum_objects():
    checksum = Buffer(b"not necessarily available").get_checksum()
    result = read_wire_directory(
        _index({"members": {"child": {"__checksum__": checksum.hex()}}}),
        "deepcell",
    )
    assert result == {"child": Checksum(checksum.hex())}
    assert type(result["child"]) is Checksum


def test_bytes_with_npy_magic_remain_bytes():
    raw = b"\x93NUMPY is only a prefix here"
    result = read_wire_directory(
        _index({"members": {"raw": "m.bytes"}}, **{"m.bytes": raw}), "folder"
    )
    assert result == {"raw": raw}
    assert type(result["raw"]) is bytes


@pytest.mark.parametrize(
    "dtype,values",
    [
        ("bool", [True, False]),
        ("int64", [1, -2]),
        ("uint64", [1, 2]),
        ("float64", [1.5, 2.5]),
    ],
)
def test_jsonarray_decoding_matches_handler_payloads(dtype, values):
    payload = {"dtype": dtype, "shape": [2], "values": values}
    without_marker = decode_jsonarray(payload)
    with_marker = decode_jsonarray({"__seamless_array__": True, **payload})
    np.testing.assert_array_equal(without_marker, with_marker)
    wire = read_wire_directory(
        _index(
            {"members": {"array": "array.jsonarray"}},
            **{"array.jsonarray": json.dumps(payload).encode()},
        ),
        "deepcell",
    )
    np.testing.assert_array_equal(wire["array"], with_marker)

    matrix = {"dtype": "int64", "shape": [2, 2], "values": [1, 3, 2, 4]}
    decoded = decode_jsonarray(matrix)
    np.testing.assert_array_equal(decoded, np.array([[1, 2], [3, 4]]))


def test_json_members_merge_and_keys_are_unreserved():
    checksum = Buffer("index member", "mixed").get_checksum()
    result = read_wire_directory(
        _index(
            {
                "members": {
                    "value": "value.bytes",
                    "json": "json.bytes",
                    "__checksum__": {"__checksum__": checksum.hex()},
                },
                "json": "small.json",
            },
            **{
                "value.bytes": b"v",
                "json.bytes": b"j",
                "small.json": b'{"small": 3}',
            },
        ),
        "folder",
    )
    assert result == {
        "value": b"v",
        "json": b"j",
        "__checksum__": checksum,
        "small": 3,
    }


@pytest.mark.parametrize(
    "index,files,celltype,entry",
    [
        ({"extra": 1, "members": {}}, {}, "deepcell", "extra"),
        ({"members": {}, "value": "v.json"}, {"v.json": b"null"}, "deepcell", "index.json"),
        ({}, {}, "deepcell", "index.json"),
        ({"members": {}}, {}, "mixed", "members"),
        ({"value": "v.json"}, {"v.json": b"null"}, "deepcell", "value"),
        ({"members": {"bad": 5}}, {}, "deepcell", "bad"),
        ({"members": {"bad": {"__checksum__": "A" * 64}}}, {}, "deepcell", "bad"),
        ({"members": {"bad": {"__checksum__": "a" * 64, "extra": 1}}}, {}, "deepcell", "bad"),
        ({"members": {"bad": "../x.bytes"}}, {}, "deepcell", "bad"),
        ({"members": {"bad": "dir/x.bytes"}}, {}, "deepcell", "bad"),
        ({"members": {"bad": "x.unknown"}}, {"x.unknown": b"x"}, "deepcell", "bad"),
        ({"members": {"bad": "missing.bytes"}}, {}, "deepcell", "bad"),
        ({"members": {}, "json": "side.json"}, {"side.json": b"[]"}, "deepcell", "json"),
        (
            {"members": {"same": "one.bytes"}, "json": "side.json"},
            {"one.bytes": b"1", "side.json": b'{"same":2}'},
            "deepcell",
            "same",
        ),
        (
            {"members": {"bad": "empty.jsonarray"}},
            {"empty.jsonarray": b"{}"},
            "deepcell",
            r"bad.*empty\.jsonarray.*missing required fields",
        ),
    ],
)
def test_malformed_index_names_offending_entry(index, files, celltype, entry):
    with pytest.raises((ValueError, TypeError), match=entry):
        read_wire_directory(_index(index, **files), celltype)


def test_write_checksum_pin_indices_roundtrip():
    checksum = Buffer("pin", "mixed").get_checksum()
    deep_index = write_deepcell_pin_index({"a": checksum})
    deep = read_wire_directory({"index.json": deep_index}, "deepcell")
    assert deep == {"a": checksum}
    assert type(deep["a"]) is Checksum

    checksum_index = write_checksum_pin_index(checksum)
    decoded = read_wire_directory({"index.json": checksum_index}, "checksum")
    assert decoded == checksum
    assert type(decoded) is Checksum


def test_mapping_decoder_never_opens_files(monkeypatch):
    def fail_open(*_args, **_kwargs):
        raise AssertionError("helper B must decode only the supplied mapping")

    monkeypatch.setattr(builtins, "open", fail_open)
    result = read_wire_directory(
        _index({"members": {"value": "member.bytes"}}, **{"member.bytes": b"ok"}),
        "folder",
    )
    assert result == {"value": b"ok"}
