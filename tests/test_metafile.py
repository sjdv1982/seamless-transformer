"""Tests for --metafile flag: file-typed meta-pins."""

import hashlib
import pytest

from seamless_transformer.cmd.bash_transformation import prepare_bash_transformation


def _fake_register_buffer(buffer, dry_run=False):
    return hashlib.sha256(bytes(buffer)).hexdigest()


def _fake_register_dict(value, dry_run=False):
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _common_kwargs(monkeypatch):
    monkeypatch.setattr(
        "seamless_transformer.cmd.bash_transformation.register_buffer",
        _fake_register_buffer,
    )
    monkeypatch.setattr(
        "seamless_transformer.cmd.bash_transformation.register_dict",
        _fake_register_dict,
    )
    return dict(
        code="cat config.txt",
        checksum_dict={},
        directories=[],
        make_executables=[],
        result_targets=None,
        capture_stdout=True,
        environment={},
        meta={},
        variables=None,
        meta_variables=None,
        dry_run=True,
    )


def test_metafile_pin_excluded_from_identity(monkeypatch):
    """Changing a meta-file checksum must not change the transformation checksum."""
    kwargs = _common_kwargs(monkeypatch)

    checksum_a = "a" * 64
    checksum_b = "b" * 64

    tf_checksum1, tf_dict1 = prepare_bash_transformation(
        **kwargs, meta_file_checksums={"config.txt": checksum_a}
    )
    tf_checksum2, tf_dict2 = prepare_bash_transformation(
        **kwargs, meta_file_checksums={"config.txt": checksum_b}
    )

    assert tf_checksum1 == tf_checksum2
    assert tf_dict1["META__FILE__config.txt"][2] != tf_dict2["META__FILE__config.txt"][2]


def test_metafile_pin_stored_with_correct_prefix(monkeypatch):
    """Meta-file pins must be stored under META__FILE__<name>."""
    kwargs = _common_kwargs(monkeypatch)
    checksum = "c" * 64

    _, tf_dict = prepare_bash_transformation(
        **kwargs, meta_file_checksums={"config.txt": checksum}
    )

    assert "META__FILE__config.txt" in tf_dict
    assert tf_dict["META__FILE__config.txt"] == ("bytes", None, checksum)


def test_metafile_none_has_no_effect(monkeypatch):
    """meta_file_checksums=None must be equivalent to not passing it."""
    kwargs = _common_kwargs(monkeypatch)

    tf_checksum1, _ = prepare_bash_transformation(**kwargs, meta_file_checksums=None)
    tf_checksum2, _ = prepare_bash_transformation(**kwargs)

    assert tf_checksum1 == tf_checksum2
