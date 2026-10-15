"""Phase 3 contract tests for bash directory pins and deep results."""

import gzip
import os
from pathlib import Path
import subprocess

import pytest

from seamless import Buffer, CacheMissError, Checksum
from seamless_transformer import transformation_namespace
from seamless_transformer.execute_bash import execute_bash, write_directory_pin


UBUNTU = "ubuntu@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401"


def _place_buffer(data, directory, suffix=""):
    buffer = Buffer(data)
    checksum = buffer.get_checksum()
    path = Path(directory) / checksum.hex()[:2] / (checksum.hex() + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.content)
    return checksum, path


def _use_read_directory(monkeypatch, directory):
    monkeypatch.setattr(
        transformation_namespace,
        "_get_read_folder_directories",
        lambda: [str(directory)],
    )


def _bash(code, pins, filesystem=None, output="text", **kwargs):
    return execute_bash(
        code,
        list(pins),
        "",
        pins,
        filesystem or {},
        output,
        **kwargs,
    )


@pytest.fixture(scope="session", params=["docker", "podman", "apptainer"])
def directory_pin_image(request, tmp_path_factory):
    runtime = request.param
    if runtime in ("docker", "podman"):
        subprocess.run([runtime, "info"], check=True, capture_output=True, timeout=180)
        inspected = subprocess.run(
            [runtime, "image", "inspect", UBUNTU], capture_output=True, timeout=180
        )
        if inspected.returncode:
            subprocess.run([runtime, "pull", UBUNTU], check=True, timeout=300)
        return runtime, UBUNTU, None

    subprocess.run(["apptainer", "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("phase3-bash-apptainer")
    image_path = directory / (UBUNTU + ".sif")
    result = subprocess.run(
        ["apptainer", "pull", str(image_path), "docker://" + UBUNTU],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return runtime, UBUNTU, directory


@pytest.fixture
def container_settings(directory_pin_image, monkeypatch):
    runtime, image, apptainer_dir = directory_pin_image
    monkeypatch.setenv("SEAMLESS_CONTAINER_RUNTIME", runtime)
    monkeypatch.setenv("SEAMLESS_CONTAINER_PULL", "never")
    monkeypatch.delenv("SEAMLESS_CONTAINER_ARGS", raising=False)
    if apptainer_dir is not None:
        monkeypatch.setenv("SEAMLESS_APPTAINER_DIR", str(apptainer_dir))
    return runtime, image


def test_directory_pin_creates_new_directory_and_links_disk_members(tmp_path):
    read_dir = tmp_path / "buffers"
    checksum, stored = _place_buffer(b"visible\n", read_dir)
    job = tmp_path / "job"
    pin = job / "data"

    write_directory_pin(
        pin,
        {"nested/member.txt": checksum},
        "folder",
        read_directories=[str(read_dir)],
    )

    assert pin.is_dir()
    linked = pin / "nested/member.txt"
    assert linked.is_symlink()
    assert linked.resolve() == stored.resolve()
    assert linked.read_bytes() == b"visible\n"


def test_folder_remote_member_is_written(monkeypatch, tmp_path):
    value = b"served only by the buffer source\n"
    checksum = Buffer(value).get_checksum()
    buffer = Buffer(value)
    calls = []

    def resolve(self):
        calls.append(self.hex())
        assert self == checksum
        return buffer

    monkeypatch.setattr(Checksum, "resolve", resolve)
    pin = tmp_path / "folder"
    write_directory_pin(pin, {"remote.txt": checksum}, "folder", read_directories=[])

    member = pin / "remote.txt"
    assert calls == [checksum.hex()]
    assert member.read_bytes() == value
    assert not member.is_symlink()


def test_bash_deepfolder_pin_is_refused_without_resolving_members(monkeypatch, tmp_path):
    checksum = Buffer(b"not stored").get_checksum()

    def no_resolve(self):
        raise AssertionError("a refused deepfolder pin must not resolve members")

    monkeypatch.setattr(Checksum, "resolve", no_resolve)
    with pytest.raises(TypeError, match="cannot be deepfolder"):
        write_directory_pin(
            tmp_path / "deep",
            {"remote.txt": checksum},
            "deepfolder",
            read_directories=[],
        )
    assert not (tmp_path / "deep").exists()


def test_bash_deepfolder_pin_is_refused_regardless_of_storage_form(tmp_path):
    read_dir = tmp_path / "buffers"
    data = b"compressed deep member\n"
    checksum = Buffer(data).get_checksum()
    stored = read_dir / checksum.hex()[:2] / (checksum.hex() + ".gz")
    stored.parent.mkdir(parents=True)
    stored.write_bytes(gzip.compress(data))
    pin = tmp_path / "deep"

    with pytest.raises(TypeError, match="cannot be deepfolder"):
        write_directory_pin(
            pin,
            {"member.txt": checksum},
            "deepfolder",
            read_directories=[str(read_dir)],
        )
    assert not pin.exists()


def test_folder_compression_follows_pin_name(tmp_path):
    read_dir = tmp_path / "buffers"
    data = b"compressed folder member\n"
    checksum = Buffer(data).get_checksum()
    stored = read_dir / checksum.hex()[:2] / (checksum.hex() + ".gz")
    stored.parent.mkdir(parents=True)
    compressed = gzip.compress(data)
    stored.write_bytes(compressed)

    uncompressed_pin = tmp_path / "folder"
    write_directory_pin(
        uncompressed_pin,
        {"member.txt": checksum},
        "folder",
        read_directories=[str(read_dir)],
    )
    assert (uncompressed_pin / "member.txt").read_bytes() == data
    assert not (uncompressed_pin / "member.txt").is_symlink()

    compressed_pin = tmp_path / "folder.gz"
    write_directory_pin(
        compressed_pin,
        {"member.txt": checksum},
        "folder",
        read_directories=[str(read_dir)],
        compression_suffix=".gz",
    )
    member = compressed_pin / "member.txt.gz"
    assert member.is_symlink()
    assert member.resolve() == stored.resolve()


def test_container_reads_linked_and_written_members(
    container_settings, monkeypatch, tmp_path
):
    _runtime, image = container_settings
    read_dir = tmp_path / "buffers"
    linked_checksum, _ = _place_buffer(b"linked\n", read_dir)
    remote_value = b"written\n"
    remote_buffer = Buffer(remote_value)
    remote_checksum = remote_buffer.get_checksum()
    original_resolve = Checksum.resolve

    def resolve(self):
        if self == remote_checksum:
            return remote_buffer
        return original_resolve(self)

    monkeypatch.setattr(Checksum, "resolve", resolve)
    _use_read_directory(monkeypatch, read_dir)
    result = _bash(
        "cat data/linked data/remote > RESULT",
        {"data": {"linked": linked_checksum, "remote": remote_checksum}},
        {"data": {"mode": "directory", "filesystem": False, "celltype": "folder"}},
        docker_={"name": image},
    )
    assert result == "linked\nwritten\n"


def test_external_buffer_symlink_without_container(monkeypatch, tmp_path):
    read_dir = tmp_path / "buffers"
    checksum, stored = _place_buffer(b"external target\n", read_dir)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(stored.read_bytes())
    stored.unlink()
    stored.symlink_to(outside)
    _use_read_directory(monkeypatch, read_dir)

    with pytest.raises(TypeError, match="cannot be deepfolder"):
        _bash(
            "cat data/member > RESULT",
            {"data": {"member": checksum}},
            {"data": {"mode": "directory", "filesystem": False, "celltype": "deepfolder"}},
            output="text",
        )


def test_container_external_buffer_symlink_fallback(
    container_settings, monkeypatch, tmp_path
):
    _runtime, image = container_settings
    read_dir = tmp_path / "buffers"
    checksum, stored = _place_buffer(b"copied into the job\n", read_dir)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(stored.read_bytes())
    stored.unlink()
    stored.symlink_to(outside)
    _use_read_directory(monkeypatch, read_dir)

    result = _bash(
        "cat data/member > RESULT",
        {"data": {"member": checksum}},
        {"data": {"mode": "directory", "filesystem": False, "celltype": "folder"}},
        docker_={"name": image},
    )
    assert result == "copied into the job\n"


def test_container_internal_symlink_and_hardlink(
    container_settings, monkeypatch, tmp_path
):
    _runtime, image = container_settings
    read_dir = tmp_path / "buffers"
    symlink_checksum, symlink_path = _place_buffer(b"internal symlink\n", read_dir)
    hardlink_checksum, hardlink_path = _place_buffer(b"internal hardlink\n", read_dir)
    symlink_target = read_dir / "inside-symlink-target"
    hardlink_target = read_dir / "inside-hardlink-target"
    symlink_target.write_bytes(symlink_path.read_bytes())
    hardlink_target.write_bytes(hardlink_path.read_bytes())
    symlink_path.unlink()
    symlink_path.symlink_to(symlink_target)
    hardlink_path.unlink()
    os.link(hardlink_target, hardlink_path)
    _use_read_directory(monkeypatch, read_dir)

    result = _bash(
        "cat data/symlink data/hardlink > RESULT",
        {"data": {"symlink": symlink_checksum, "hardlink": hardlink_checksum}},
        {"data": {"mode": "directory", "filesystem": False, "celltype": "folder"}},
        docker_={"name": image},
    )
    assert result == "internal symlink\ninternal hardlink\n"


@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder"])
def test_bash_deep_pins_refused_before_launch(tmp_path, celltype):
    from seamless_transformer import Transformer

    sentinel = tmp_path / "ran"
    transformer = Transformer("bash", direct=True)
    transformer.code = f"touch {sentinel}"
    with pytest.raises(TypeError, match=f"cannot be {celltype}.*folder or plain"):
        transformer.celltypes.data = celltype
    assert not sentinel.exists()


@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder"])
@pytest.mark.parametrize("checksum", ["a" * 64, None], ids=["set", "absent"])
def test_bash_deep_pins_are_refused_when_celltype_validation_is_bypassed(
    celltype, checksum
):
    from seamless_transformer.transformation_namespace import (
        build_transformation_namespace_sync,
    )

    transformation = {
        "__language__": "bash",
        "__output__": ("plain", "result"),
        "data": (celltype, None, checksum),
    }
    with pytest.raises(TypeError, match=f"cannot be {celltype}.*folder or plain"):
        build_transformation_namespace_sync(transformation)


def test_bash_deepcell_result_preserves_file_bytes():
    expected = {"compact.json": b'{"a":1}', "npy-like": b"3"}
    result = _bash(
        "mkdir RESULT; printf '%s' '{\"a\":1}' > RESULT/compact.json; "
        "printf 3 > RESULT/npy-like",
        {},
        output="deepcell",
    )
    assert result == expected


def test_bash_deepcell_result_rejects_invalid_member():
    with pytest.raises(Exception, match="bad.txt"):
        _bash(
            "mkdir RESULT; printf 'hello' > RESULT/bad.txt",
            {},
            output="deepcell",
        )


@pytest.mark.parametrize("celltype", ["folder", "deepcell"])
@pytest.mark.parametrize("body", ["printf value > RESULT", "mkdir RESULT"])
def test_bash_deep_result_requires_nonempty_directory(celltype, body):
    with pytest.raises(Exception, match="requires RESULT to be a directory|empty"):
        _bash(body, {}, output=celltype)
