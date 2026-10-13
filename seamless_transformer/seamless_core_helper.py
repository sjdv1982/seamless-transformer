"""Seamless-core-dependent helpers staged only for Python CLI jobs."""

from __future__ import annotations

import json
from collections.abc import Mapping

from seamless import Checksum
from seamless.util.mixed.io import deserialize, serialize


def decode_mixed(data):
    return deserialize(data)[0]


def encode_mixed(value):
    return serialize(value)


def decode_checksum_pin(data, celltype):
    try:
        index = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed {celltype} pin index: {exc}") from exc
    if not isinstance(index, Mapping):
        raise ValueError(f"Malformed {celltype} pin index: expected an object")

    if celltype == "checksum":
        entry = index.get("value")
        if not isinstance(entry, Mapping) or set(entry) != {"__checksum__"}:
            raise ValueError("Malformed checksum pin index: expected one checksum")
        value = entry["__checksum__"]
        if not isinstance(value, str):
            raise ValueError("Malformed checksum pin index: checksum must be text")
        return Checksum(value)

    members = index.get("members")
    if not isinstance(members, Mapping):
        raise ValueError("Malformed deepcell pin index: expected members")
    result = {}
    for key, entry in members.items():
        if not isinstance(key, str):
            raise ValueError("Malformed deepcell pin index: keys must be strings")
        if not isinstance(entry, Mapping) or set(entry) != {"__checksum__"}:
            raise ValueError(f"Malformed deepcell pin index member {key!r}")
        value = entry["__checksum__"]
        if not isinstance(value, str):
            raise ValueError(f"Malformed deepcell checksum {key!r}: expected text")
        result[key] = Checksum(value)
    return result


__all__ = ["decode_checksum_pin", "decode_mixed", "encode_mixed"]
