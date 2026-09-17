import gc
import weakref

import pytest

from seamless import Buffer
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.reference_lifecycle import collect_refholder_claims, safe_release_refholder
from seamless_transformer.pretransformation import PreTransformation
from seamless_transformer.code_manager import CodeManager


def test_pretransformation_non_scratch_input_is_a_refholder():
    checksum = Buffer(b"pretransformation input").get_checksum()
    pre = PreTransformation(
        {
            "__language__": "python",
            "__output__": ("result", "mixed", None),
            "value": ("text", None, checksum),
        }
    )
    pre.prepare_transformation()
    assert len(pre._value_refs) == 1
    assert get_buffer_cache().reference_snapshot()[checksum][0] == 1
    pre.release()
    pre.release()
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0


def test_pretransformation_scratch_input_is_tempref_only():
    checksum = Buffer(b"pretransformation scratch").get_checksum()
    checksum.tempref(scratch=True)
    pre = PreTransformation(
        {
            "__language__": "python",
            "__output__": ("result", "mixed", None),
            "value": ("text", None, checksum),
        }
    )
    pre.prepare_transformation()
    assert pre._value_refs == []
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0
    pre.release()


def test_mixed_scratch_and_non_scratch_pins_have_exact_roles():
    scratch_buffer = Buffer(b"mixed-scratch")
    scratch = scratch_buffer.get_checksum()
    scratch.tempref(scratch=True)
    normal_buffer = Buffer(b"mixed-normal")
    normal = normal_buffer.get_checksum()
    pre = PreTransformation(
        {
            "__language__": "python",
            "__output__": ("result", "mixed", None),
            "scratch": ("bytes", None, scratch),
            "normal": ("bytes", None, normal),
        }
    )
    pre.prepare_transformation()
    assert pre._value_refs == [(normal, "input:normal")]
    assert collect_refholder_claims([pre]).get(scratch, []) == []
    assert collect_refholder_claims([pre])[normal][0][1] == "input:normal"
    assert get_buffer_cache().reference_snapshot().get(scratch, (0, 0, False))[0] == 0
    assert get_buffer_cache().reference_snapshot()[normal][0] == 1
    pre.release()
    assert get_buffer_cache().reference_snapshot().get(normal, (0, 0, False))[0] == 0


def test_repeated_checksum_pins_are_repeated_claims_and_counts():
    source = Buffer(b"repeated-pre-pin")
    checksum = source.get_checksum()
    pre = PreTransformation(
        {
            "__language__": "python",
            "__output__": ("result", "mixed", None),
            "first": ("bytes", None, checksum),
            "second": ("bytes", None, checksum),
        }
    )
    pre.prepare_transformation()
    assert pre._value_refs == [
        (checksum, "input:first"),
        (checksum, "input:second"),
    ]
    assert get_buffer_cache().reference_snapshot()[checksum][0] == 2
    assert [role for _, role in collect_refholder_claims([pre])[checksum]] == [
        "input:first",
        "input:second",
    ]
    pre.release()
    pre.release()
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0


def test_partial_preparation_failure_releases_converted_prefix():
    source = Buffer(b"partial-preparation")
    checksum = source.get_checksum()
    pre = PreTransformation(
        {
            "__language__": "python",
            "__output__": ("result", "mixed", None),
            "first": ("bytes", None, checksum),
            "bad": ("does-not-exist", None, object()),
        }
    )
    with pytest.raises((TypeError, ValueError)):
        pre.prepare_transformation()
    assert pre._refholds_released is True
    assert not pre._value_refs
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0


def test_code_manager_claims_are_not_attributed_to_pretransformation():
    manager = CodeManager()
    code = Buffer("result = value", "python")
    pre = PreTransformation(
        {
            "__language__": "python",
            "__output__": ("result", "mixed", None),
            "code": ("python", None, code),
        },
        code_manager=manager,
    )
    pre.prepare_transformation()
    semantic, syntactic = manager.track_code_buffer(code)
    assert all(checksum not in {semantic, syntactic} for checksum, _ in pre._refheld_checksums())
    manager_claims = collect_refholder_claims([manager])
    assert manager_claims[semantic]
    assert manager_claims[syntactic]
    pre.release()
    manager._release_refholds()


def test_transfer_acquires_before_pretransformation_release(monkeypatch):
    source = Buffer(b"transfer-order")
    checksum = source.get_checksum()
    checksum.tempref(scratch=True)
    pre = PreTransformation(
        {
            "__language__": "python",
            "__output__": ("result", "mixed", None),
            "value": ("bytes", None, checksum),
        }
    )
    pre.prepare_transformation()
    from seamless_transformer.transformation_class import transformation_from_pretransformation

    snapshots = []
    original_incref = type(checksum).__dict__["incref_refholder"]
    original_decref = type(checksum).__dict__["decref_refholder"]

    def record_incref(*args, **kwargs):
        result = original_incref(args[0], **kwargs)
        snapshots.append(("acquire", get_buffer_cache().reference_snapshot()[checksum]))
        return result

    def record_decref(*args, **kwargs):
        result = original_decref(args[0], **kwargs)
        snapshots.append(("release", get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))))
        return result

    monkeypatch.setattr(type(checksum), "incref_refholder", record_incref)
    monkeypatch.setattr(type(checksum), "decref_refholder", record_decref)
    transformation = transformation_from_pretransformation(
        pre, upstream_dependencies={}, meta={}, scratch=False
    )
    assert any(kind == "acquire" and snapshot[2] for kind, snapshot in snapshots)
    assert all(
        not (kind == "release" and snapshot[0] == 0)
        for kind, snapshot in snapshots
    )
    transformation._release_refholds()


def test_safe_finalizer_reports_cleanup_exception(caplog):
    class BrokenHolder:
        def _release_refholds(self):
            raise RuntimeError("broken pre cleanup")

    holder = BrokenHolder()
    with caplog.at_level("WARNING", logger="seamless.references"):
        safe_release_refholder(holder)
    assert "BrokenHolder" in caplog.text
    assert "broken pre cleanup" in caplog.text
