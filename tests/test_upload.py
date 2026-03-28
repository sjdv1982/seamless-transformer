from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import seamless_config.cluster as cluster_mod
import seamless_config.extern_clients as extern_clients
import seamless_config.select as select
from seamless.checksum.calculate_checksum import calculate_checksum
from seamless import Checksum
from seamless_transformer.cmd.api import upload
from seamless_transformer.cmd.register import (
    DestinationEntryError,
    _resolve_destination_path,
    register_buffer,
    register_file,
)


def _raise_runtime_error(*args):
    raise RuntimeError(" ".join(str(arg) for arg in args))


def _make_source_file(tmp_path, name="input.txt", content=b"hello world"):
    path = tmp_path / name
    path.write_bytes(content)
    return path, calculate_checksum(content)


def test_upload_manual_destination_bypasses_seamless_config(monkeypatch, tmp_path):
    source, _ = _make_source_file(tmp_path)
    destination = tmp_path / "buffers"
    destination.mkdir()

    def fail(*args, **kwargs):
        raise AssertionError("seamless-config should be bypassed in manual mode")

    captured = {}

    def fake_files_to_checksums(paths, **kwargs):
        captured["paths"] = list(paths)
        captured["kwargs"] = kwargs
        return {str(source): "0" * 64}, {}

    monkeypatch.setattr(upload.seamless_config, "init", fail)
    monkeypatch.setattr(upload.seamless_config, "set_stage", fail)
    monkeypatch.setattr(extern_clients, "set_remote_clients_from_env", fail)
    monkeypatch.setattr(upload, "files_to_checksums", fake_files_to_checksums)

    assert (
        upload._main(["--destination", str(destination), str(source)]) is None
    )
    assert captured["paths"] == [str(source)]
    assert captured["kwargs"]["destination_folder"] == str(destination)
    assert captured["kwargs"]["transfer_mode"] == "copy"
    assert captured["kwargs"]["existing_entry_policy"] == "skip"
    checksum_file = source.with_suffix(source.suffix + ".CHECKSUM")
    assert checksum_file.read_text().strip() == "0" * 64


@pytest.mark.parametrize("flag", ["--project", "--stage"])
def test_upload_manual_destination_rejects_config_flags(monkeypatch, tmp_path, flag):
    source, _ = _make_source_file(tmp_path)
    destination = tmp_path / "buffers"
    destination.mkdir()
    monkeypatch.setattr(upload, "err", _raise_runtime_error)

    with pytest.raises(RuntimeError, match="--destination cannot be combined"):
        upload._main([flag, "demo", "--destination", str(destination), str(source)])


def test_upload_auto_destination_defaults_to_copy_and_dest_skip(
    monkeypatch, tmp_path
):
    source, _ = _make_source_file(tmp_path)
    destination = tmp_path / "buffers"
    destination.mkdir()

    captured = {}

    def fake_files_to_checksums(paths, **kwargs):
        captured["paths"] = list(paths)
        captured["kwargs"] = kwargs
        return {str(source): "1" * 64}, {}

    monkeypatch.setattr(
        extern_clients, "set_remote_clients_from_env", lambda include_dask: False
    )
    monkeypatch.setattr(upload.seamless_config, "init", lambda workdir: None)
    monkeypatch.setattr(select, "get_selected_cluster", lambda: "demo")
    monkeypatch.setattr(
        cluster_mod, "get_cluster", lambda cluster: SimpleNamespace(type="local")
    )
    monkeypatch.setattr(
        upload, "_get_auto_destination_folder", lambda: str(destination)
    )
    monkeypatch.setattr(upload, "files_to_checksums", fake_files_to_checksums)

    assert upload._main([str(source)]) is None
    assert captured["paths"] == [str(source)]
    assert captured["kwargs"]["destination_folder"] == str(destination)
    assert captured["kwargs"]["transfer_mode"] == "copy"
    assert captured["kwargs"]["existing_entry_policy"] == "skip"


def test_upload_explicit_direct_mode_requires_local_cluster_or_destination(
    monkeypatch, tmp_path
):
    source, _ = _make_source_file(tmp_path)
    monkeypatch.setattr(upload, "err", _raise_runtime_error)
    monkeypatch.setattr(
        extern_clients, "set_remote_clients_from_env", lambda include_dask: False
    )
    monkeypatch.setattr(upload.seamless_config, "init", lambda workdir: None)
    monkeypatch.setattr(select, "get_selected_cluster", lambda: "remote")
    monkeypatch.setattr(
        cluster_mod, "get_cluster", lambda cluster: SimpleNamespace(type="slurm")
    )

    with pytest.raises(
        RuntimeError,
        match="Direct destination mode requires a selected local cluster or --destination",
    ):
        upload._main(["--copy", str(source)])


@pytest.mark.parametrize("prefix_layout", [False, True])
@pytest.mark.parametrize("transfer_mode", ["copy", "hardlink", "symlink"])
def test_register_file_direct_modes_create_expected_entries(
    tmp_path, transfer_mode, prefix_layout
):
    source, checksum_hex = _make_source_file(tmp_path, content=b"payload")
    destination = tmp_path / "buffers"
    destination.mkdir()
    if prefix_layout:
        (destination / ".HASHSERVER_PREFIX").write_text("")

    register_file(
        str(source),
        destination_folder=str(destination),
        transfer_mode=transfer_mode,
    )

    target = _resolve_destination_path(
        str(destination), checksum_hex, create_dirs=False
    )
    assert Checksum(checksum_hex)
    if transfer_mode == "copy":
        assert not target.endswith(".LOCK")
        assert not os.path.islink(target)
        assert open(target, "rb").read() == b"payload"
        assert open(target, "rb").read() == source.read_bytes()
        assert os.stat(target).st_ino != os.stat(source).st_ino
    elif transfer_mode == "hardlink":
        assert not os.path.islink(target)
        assert os.stat(target).st_ino == os.stat(source).st_ino
    else:
        assert os.path.islink(target)
        assert os.readlink(target) == str(source.resolve())


def test_register_file_dest_skip_trusts_existing_entry(tmp_path):
    source, checksum_hex = _make_source_file(tmp_path, content=b"good")
    destination = tmp_path / "buffers"
    destination.mkdir()
    target = _resolve_destination_path(str(destination), checksum_hex)
    with open(target, "wb") as f:
        f.write(b"bad")

    register_file(
        str(source),
        destination_folder=str(destination),
        transfer_mode="copy",
        existing_entry_policy="skip",
    )

    assert open(target, "rb").read() == b"bad"


def test_register_file_dest_verify_accepts_valid_and_rejects_corrupt(tmp_path):
    source, checksum_hex = _make_source_file(tmp_path, content=b"good")
    destination = tmp_path / "buffers"
    destination.mkdir()
    target = _resolve_destination_path(str(destination), checksum_hex)

    with open(target, "wb") as f:
        f.write(b"good")
    register_file(
        str(source),
        destination_folder=str(destination),
        transfer_mode="copy",
        existing_entry_policy="verify",
    )
    assert open(target, "rb").read() == b"good"

    with open(target, "wb") as f:
        f.write(b"bad")
    with pytest.raises(DestinationEntryError, match="does not match checksum"):
        register_file(
            str(source),
            destination_folder=str(destination),
            transfer_mode="copy",
            existing_entry_policy="verify",
        )


def test_register_file_dest_repair_replaces_corrupt_entry_with_regular_file(tmp_path):
    source, checksum_hex = _make_source_file(tmp_path, content=b"good")
    destination = tmp_path / "buffers"
    destination.mkdir()
    target = _resolve_destination_path(str(destination), checksum_hex)
    target_symlink = tmp_path / "wrong-target"
    target_symlink.write_bytes(b"bad")
    target_path = target
    if os.path.lexists(target_path):
        os.unlink(target_path)
    os.symlink(str(target_symlink), target_path)

    register_file(
        str(source),
        destination_folder=str(destination),
        transfer_mode="symlink",
        existing_entry_policy="repair",
    )

    assert not os.path.islink(target_path)
    assert open(target_path, "rb").read() == b"good"


@pytest.mark.parametrize(
    ("existing_entry_policy", "expected"),
    [("skip", "skip"), ("verify", "error"), ("repair", "repair")],
)
def test_register_buffer_existing_entry_policies_for_directory_buffers(
    tmp_path, existing_entry_policy, expected
):
    buffer = b'{"file.txt": "abc"}\n'
    checksum_hex = calculate_checksum(buffer)
    destination = tmp_path / "buffers"
    destination.mkdir()
    target = _resolve_destination_path(str(destination), checksum_hex)
    with open(target, "wb") as f:
        f.write(b"wrong")

    if expected == "error":
        with pytest.raises(DestinationEntryError, match="does not match checksum"):
            register_buffer(
                buffer,
                destination_folder=str(destination),
                transfer_mode="hardlink",
                existing_entry_policy=existing_entry_policy,
            )
        assert open(target, "rb").read() == b"wrong"
        return

    register_buffer(
        buffer,
        destination_folder=str(destination),
        transfer_mode="hardlink",
        existing_entry_policy=existing_entry_policy,
    )

    if expected == "skip":
        assert open(target, "rb").read() == b"wrong"
    else:
        assert open(target, "rb").read() == buffer
