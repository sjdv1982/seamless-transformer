"""Contract test: contracts/internal/checksum-reference-lifecycle.md §8 *Deep
checksums* — a Transformation that produces a deep result claims the index
checksum under its "result" role and no claim on the leaves."""

from __future__ import annotations

import uuid

import pytest

from seamless import Checksum
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.reference_lifecycle import collect_refholder_claims
from seamless.transformer import delayed


def produce_deepcell(token):
    return {"left": "left-" + token, "right": "right-" + token}


def produce_folder(token):
    return {"left": ("left-" + token).encode(), "right": ("right-" + token).encode()}


PRODUCERS = {"deepcell": produce_deepcell, "folder": produce_folder}


@pytest.mark.parametrize("celltype", ["deepcell", "folder"])
def test_deep_result_claims_only_the_index(celltype):
    tf_builder = delayed(PRODUCERS[celltype])
    tf_builder.local = True
    tf_builder.celltypes.result = celltype
    transformation = tf_builder(uuid.uuid4().hex)
    result = transformation.compute()
    assert result is not None, transformation.exception
    cache = get_buffer_cache()
    try:
        roles = [role for cs, role in transformation._refheld_checksums() if cs == result]
        assert roles == ["result"]
        assert cache.reference_snapshot()[result][0] == 1
        index = result.resolve(celltype)
        assert len(index) == 2
        claims = collect_refholder_claims()
        for member in index.values():
            member = Checksum(member)
            assert cache.reference_snapshot().get(member, (0, 0, False))[0] == 0
            assert member not in claims
    finally:
        transformation._release_refholds()
    assert cache.reference_snapshot().get(result, (0, 0, False))[0] == 0
