import asyncio

import pytest
from seamless import Checksum
from seamless_remote import database_remote

from seamless_transformer import transformation_class
from seamless_transformer.cmd import bash_transformation


TF_CHECKSUM = Checksum("01" * 32)
RESULT_CHECKSUM = Checksum("02" * 32)


class _FakeTransformation:
    transformation_checksum = TF_CHECKSUM
    exception = None

    async def computation(self, *, require_value):
        return RESULT_CHECKSUM


def _patch_transformation(monkeypatch):
    transformation = _FakeTransformation()
    monkeypatch.setattr(
        transformation_class,
        "transformation_from_dict",
        lambda *args, **kwargs: transformation,
    )
    return transformation


def test_sync_undo_reports_irreproducible_result_without_clearing(monkeypatch):
    transformation = _patch_transformation(monkeypatch)
    reported = []

    def compute(_transformation, *, require_value):
        return RESULT_CHECKSUM

    async def report(tf_checksum, result_checksum):
        reported.append((tf_checksum, result_checksum))
        return True

    monkeypatch.setattr(transformation_class, "compute_transformation_sync", compute)
    monkeypatch.setattr(database_remote, "report_irreproducible_result", report)

    result = bash_transformation.run_transformation({}, undo=True)

    assert result == RESULT_CHECKSUM
    assert reported == [(transformation.transformation_checksum, RESULT_CHECKSUM)]


def test_async_undo_reports_irreproducible_result(monkeypatch):
    transformation = _patch_transformation(monkeypatch)
    reported = []

    async def report(tf_checksum, result_checksum):
        reported.append((tf_checksum, result_checksum))
        return True

    monkeypatch.setattr(database_remote, "report_irreproducible_result", report)

    result = asyncio.run(
        bash_transformation.run_transformation_async({}, undo=True)
    )

    assert result == RESULT_CHECKSUM
    assert reported == [(transformation.transformation_checksum, RESULT_CHECKSUM)]


def test_undo_fails_when_database_refuses_irreproducible_report(monkeypatch):
    _patch_transformation(monkeypatch)

    def compute(_transformation, *, require_value):
        return RESULT_CHECKSUM

    async def report(_tf_checksum, _result_checksum):
        return False

    monkeypatch.setattr(transformation_class, "compute_transformation_sync", compute)
    monkeypatch.setattr(database_remote, "report_irreproducible_result", report)

    with pytest.raises(RuntimeError, match="could not be declared irreproducible"):
        bash_transformation.run_transformation({}, undo=True)
