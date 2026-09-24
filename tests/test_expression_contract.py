from __future__ import annotations

import asyncio

import pytest

from seamless import Buffer, Checksum
from seamless.transformer import delayed
from seamless_transformer import transformation_utils
from seamless_transformer import worker


def test_module_is_not_a_deep_celltype():
    assert transformation_utils.DEEP_CELLTYPES == (
        "deepcell",
        "deepfolder",
        "folder",
    )
    assert transformation_utils.is_deep_celltype("module") is False


def test_transformer_result_admits_only_the_producible_deep_celltypes():
    @delayed
    def produce():
        return {}

    produce.celltypes.result = "deepcell"
    produce.celltypes.result = "folder"
    for forbidden in ("deepfolder", "module"):
        with pytest.raises(TypeError, match=forbidden):
            produce.celltypes.result = forbidden


def test_dask_dispatch_carries_the_requesters_scratch_decision(monkeypatch):
    from seamless_dask import transformer_client

    result = Checksum("e" * 64)
    observed = []

    class ExpressionFuture:
        key = "expression-contract"

        def release(self):
            pass

    class ThinFuture:
        def result(self):
            return result.hex(), None

        def release(self):
            pass

    class Scheduler:
        def submit(self, *args, **kwargs):
            return ThinFuture()

    class Client:
        client = Scheduler()

        def get_fat_checksum_future(self, checksum):
            return object()

        def get_expression_future(self, expression, input_future):
            observed.append(expression.scratch)
            return ExpressionFuture()

    monkeypatch.setattr(
        transformer_client, "get_seamless_dask_client", lambda: Client()
    )

    actual = asyncio.run(
        worker.dispatch_expression(
            Checksum("d" * 64),
            "value",
            "plain",
            "str",
            scratch=True,
        )
    )

    assert actual == result
    assert observed == [True]


def test_deepcell_packing_accepts_dict_member_values():
    member = {"inner": "mixed member value"}
    packed = transformation_utils.pack_deep_structure(
        {"outer": member}, "deepcell"
    )
    assert packed == {
        "outer": Buffer(member, "mixed").get_checksum().hex()
    }


@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder", "folder"])
def test_pin_unpacking_rejects_nested_deep_structures(celltype):
    child = Buffer("child", "mixed" if celltype == "deepcell" else "bytes")
    child.tempref()

    with pytest.raises(ValueError, match="outer|nested"):
        transformation_utils.unpack_deep_structure(
            {"outer": {"inner": child.get_checksum().hex()}}, celltype
        )
