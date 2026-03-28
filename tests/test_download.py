from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import seamless_config.cluster as cluster_mod
import seamless_config.extern_clients as extern_clients
import seamless_config.select as select
from seamless import Checksum
from seamless.checksum.calculate_checksum import calculate_checksum
from seamless_transformer.cmd import download as download_cmd
from seamless_transformer.cmd.api import download as download_api
from seamless_transformer.cmd.exceptions import SeamlessSystemExit
from seamless_transformer.cmd.register import _resolve_destination_path


def _raise_runtime_error(*args):
    raise RuntimeError(" ".join(str(arg) for arg in args))


def _write_bufferdir_file(bufferdir, content: bytes) -> str:
    checksum = calculate_checksum(content)
    path = _resolve_destination_path(str(bufferdir), checksum)
    with open(path, "wb") as f:
        f.write(content)
    return checksum


def test_download_manual_destination_bypasses_seamless_config(monkeypatch, tmp_path):
    source_directory = tmp_path / "buffers"
    source_directory.mkdir()
    checksum = _write_bufferdir_file(source_directory, b"payload")
    target = tmp_path / "result.txt"
    (target.with_suffix(target.suffix + ".CHECKSUM")).write_text(checksum + "\n")

    def fail(*args, **kwargs):
        raise AssertionError("seamless-config should be bypassed in manual mode")

    captured = {}

    def fake_download(files, directories, **kwargs):
        captured["files"] = list(files)
        captured["directories"] = list(directories)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(download_api.seamless_config, "init", fail)
    monkeypatch.setattr(download_api.seamless_config, "set_stage", fail)
    monkeypatch.setattr(extern_clients, "set_remote_clients_from_env", fail)
    monkeypatch.setattr(download_api, "download", fake_download)

    assert (
        download_api._main(["--destination", str(source_directory), str(target)])
        is None
    )
    assert captured["files"] == [str(target)]
    assert captured["directories"] == []
    assert captured["kwargs"]["source_directory"] == str(source_directory)
    assert captured["kwargs"]["transfer_mode"] == "copy"
    assert captured["kwargs"]["existing_entry_policy"] == "skip"


@pytest.mark.parametrize("flag", ["--project", "--stage"])
def test_download_manual_destination_rejects_config_flags(
    monkeypatch, tmp_path, flag
):
    source_directory = tmp_path / "buffers"
    source_directory.mkdir()
    target = tmp_path / "result.txt"
    (target.with_suffix(target.suffix + ".CHECKSUM")).write_text("0" * 64 + "\n")
    monkeypatch.setattr(download_api, "err", _raise_runtime_error)

    with pytest.raises(RuntimeError, match="--destination cannot be combined"):
        download_api._main(
            [flag, "demo", "--destination", str(source_directory), str(target)]
        )


def test_download_auto_destination_defaults_to_copy_and_dest_skip(
    monkeypatch, tmp_path
):
    target = tmp_path / "result.txt"
    (target.with_suffix(target.suffix + ".CHECKSUM")).write_text("1" * 64 + "\n")
    source_directory = tmp_path / "buffers"
    source_directory.mkdir()
    captured = {}

    def fake_download(files, directories, **kwargs):
        captured["files"] = list(files)
        captured["directories"] = list(directories)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        extern_clients, "set_remote_clients_from_env", lambda include_dask: False
    )
    monkeypatch.setattr(download_api.seamless_config, "init", lambda workdir: None)
    monkeypatch.setattr(select, "get_selected_cluster", lambda: "demo")
    monkeypatch.setattr(
        cluster_mod, "get_cluster", lambda cluster: SimpleNamespace(type="local")
    )
    monkeypatch.setattr(
        download_api, "_get_auto_destination_folder", lambda: str(source_directory)
    )
    monkeypatch.setattr(download_api, "download", fake_download)

    assert download_api._main([str(target)]) is None
    assert captured["kwargs"]["source_directory"] == str(source_directory)
    assert captured["kwargs"]["transfer_mode"] == "copy"
    assert captured["kwargs"]["existing_entry_policy"] == "skip"


def test_download_explicit_direct_mode_requires_local_cluster_or_destination(
    monkeypatch, tmp_path
):
    target = tmp_path / "result.txt"
    (target.with_suffix(target.suffix + ".CHECKSUM")).write_text("1" * 64 + "\n")
    monkeypatch.setattr(download_api, "err", _raise_runtime_error)
    monkeypatch.setattr(
        extern_clients, "set_remote_clients_from_env", lambda include_dask: False
    )
    monkeypatch.setattr(download_api.seamless_config, "init", lambda workdir: None)
    monkeypatch.setattr(select, "get_selected_cluster", lambda: "remote")
    monkeypatch.setattr(
        cluster_mod, "get_cluster", lambda cluster: SimpleNamespace(type="slurm")
    )

    with pytest.raises(
        RuntimeError,
        match="Direct destination mode requires a selected local cluster or --destination",
    ):
        download_api._main(["--hardlink", str(target)])


@pytest.mark.parametrize("transfer_mode", ["copy", "hardlink", "symlink"])
def test_direct_download_file_modes(tmp_path, transfer_mode):
    source_directory = tmp_path / "buffers"
    source_directory.mkdir()
    checksum = _write_bufferdir_file(source_directory, b"payload")
    target = tmp_path / "result.txt"

    download_cmd.download(
        [str(target)],
        [],
        checksum_dict={str(target): checksum},
        max_download_size=10**9,
        max_download_files=100,
        auto_confirm="yes",
        source_directory=str(source_directory),
        transfer_mode=transfer_mode,
        existing_entry_policy="skip",
    )

    assert target.exists() or target.is_symlink()
    if transfer_mode == "copy":
        assert not target.is_symlink()
        assert target.read_bytes() == b"payload"
        assert os.stat(target).st_ino != os.stat(
            _resolve_destination_path(str(source_directory), checksum, create_dirs=False)
        ).st_ino
    elif transfer_mode == "hardlink":
        assert not target.is_symlink()
        assert os.stat(target).st_ino == os.stat(
            _resolve_destination_path(str(source_directory), checksum, create_dirs=False)
        ).st_ino
    else:
        assert target.is_symlink()
        assert os.readlink(target) == os.path.abspath(
            _resolve_destination_path(str(source_directory), checksum, create_dirs=False)
        )


def test_direct_download_existing_entry_policies(tmp_path):
    source_directory = tmp_path / "buffers"
    source_directory.mkdir()
    checksum = _write_bufferdir_file(source_directory, b"payload")
    target = tmp_path / "result.txt"
    target.write_bytes(b"wrong")

    download_cmd.download(
        [str(target)],
        [],
        checksum_dict={str(target): checksum},
        max_download_size=10**9,
        max_download_files=100,
        auto_confirm="yes",
        source_directory=str(source_directory),
        transfer_mode="copy",
        existing_entry_policy="skip",
    )
    assert target.read_bytes() == b"wrong"

    with pytest.raises(SeamlessSystemExit, match="does not match checksum"):
        download_cmd.download(
            [str(target)],
            [],
            checksum_dict={str(target): checksum},
            max_download_size=10**9,
            max_download_files=100,
            auto_confirm="yes",
            source_directory=str(source_directory),
            transfer_mode="copy",
            existing_entry_policy="verify",
        )

    download_cmd.download(
        [str(target)],
        [],
        checksum_dict={str(target): checksum},
        max_download_size=10**9,
        max_download_files=100,
        auto_confirm="yes",
        source_directory=str(source_directory),
        transfer_mode="copy",
        existing_entry_policy="repair",
    )
    assert target.read_bytes() == b"payload"


def test_download_directory_with_unlisted_entries_errors_by_default(
    monkeypatch, tmp_path
):
    source_directory = tmp_path / "buffers"
    source_directory.mkdir()
    checksum = _write_bufferdir_file(source_directory, b"payload")
    index_buffer = json.dumps({"wanted.txt": checksum}).encode() + b"\n"
    index_path = tmp_path / "mydir.INDEX"
    index_path.write_bytes(index_buffer)
    target_dir = tmp_path / "mydir"
    target_dir.mkdir()
    (target_dir / "extra.txt").write_text("extra")
    monkeypatch.setattr(download_api, "err", _raise_runtime_error)
    monkeypatch.setattr(download_api, "download", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="contains 1 unlisted entries"):
        download_api._main(["--destination", str(source_directory), str(index_path)])


def test_download_directory_dest_rm_removes_unlisted_entries(monkeypatch, tmp_path):
    source_directory = tmp_path / "buffers"
    source_directory.mkdir()
    checksum = _write_bufferdir_file(source_directory, b"payload")
    index_buffer = json.dumps({"wanted.txt": checksum}).encode() + b"\n"
    index_path = tmp_path / "mydir.INDEX"
    index_path.write_bytes(index_buffer)
    target_dir = tmp_path / "mydir"
    target_dir.mkdir()
    extra_file = target_dir / "extra.txt"
    extra_file.write_text("extra")
    captured = {}

    def fake_download(files, directories, **kwargs):
        captured["files"] = list(files)
        captured["directories"] = list(directories)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(download_api, "download", fake_download)

    assert (
        download_api._main(
            ["--dest-rm", "--destination", str(source_directory), str(index_path)]
        )
        is None
    )
    assert not extra_file.exists()
    assert captured["directories"] == [str(target_dir)]
