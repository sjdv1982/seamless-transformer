import json
import os
import shlex
import shutil
import signal
import selectors
import subprocess
import tempfile
import threading
import sys
from io import BytesIO
from typing import IO, cast

import numpy as np

from seamless import Buffer, Checksum

from seamless.util.mixed.get_form import get_form
from seamless.util.mount_directory import write_to_directory

from .compression_utils import (
    compress_bytes,
    decompress_bytes,
    strip_compression_suffix,
)
from .process_lock import NOTIFY_ENV, current_holder


def _write_file(pinname, data, filemode, *, directory=None):
    if pinname.startswith("/"):
        raise ValueError(f"Pin {pinname}: Absolute path is not allowed")
    path_elements = pinname.split("/")
    if ".." in path_elements:
        raise ValueError(f"Pin {pinname}: .. is not allowed")
    path = pinname if directory is None else os.path.join(directory, pinname)
    if len(path_elements) > 1:
        parent_dir = os.path.dirname(path)
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, filemode) as pinf:
        pinf.write(data)


def _is_shell_env_name(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    for char in name[1:]:
        if not (char.isalnum() or char == "_"):
            return False
    return True


def write_bash_job(
    bashcode,
    pins_,
    conda_environment_,
    PINS,
    FILESYSTEM,
    *,
    script_name="transform.sh",
    directory=None,
):
    """Materialize a bash transformation job in the supplied directory."""

    def job_path(path):
        if directory is None:
            return path
        return os.path.join(directory, path)

    META_FILE_PREFIX = "META__FILE__"

    env = os.environ.copy()
    for pin in pins_:
        if pin == "pins_":
            continue
        if pin == "bashcode":
            continue
        v = PINS[pin]
        if isinstance(v, Buffer):
            v = v.content
        elif isinstance(v, Checksum):
            v = v.hex()
        if pin.startswith(META_FILE_PREFIX):
            actual_name = pin[len(META_FILE_PREFIX):]
            _write_file(
                actual_name,
                v if isinstance(v, bytes) else v.encode(),
                "bw",
                directory=directory,
            )
            continue
        if pin in FILESYSTEM:
            if FILESYSTEM[pin]["filesystem"]:
                env[pin] = v
                pin_path = job_path(pin)
                pin_parent = os.path.dirname(pin_path)
                if len(pin_parent):
                    os.makedirs(pin_parent, exist_ok=True)
                _pin_base, pin_suffix = strip_compression_suffix(pin)
                _found_base, found_suffix = strip_compression_suffix(v)
                if found_suffix == pin_suffix:
                    os.symlink(v, pin_path)
                else:
                    with open(job_path(v), "rb") as handle:
                        data = handle.read()
                    if found_suffix is not None:
                        data = decompress_bytes(data, found_suffix)
                    if pin_suffix is not None:
                        data = compress_bytes(data, pin_suffix)
                    _write_file(pin, data, "bw", directory=directory)
                continue
            elif FILESYSTEM[pin]["mode"] == "directory":
                fs_entry = FILESYSTEM[pin]
                deep = fs_entry.get("celltype") == "deepfolder"
                directory_pin, compression_suffix = strip_compression_suffix(pin)
                write_to_directory(
                    job_path(directory_pin),
                    v,
                    cleanup=False,
                    deep=deep,
                    text_only=False,
                    compression_suffix=compression_suffix,
                )
                env[pin] = directory_pin
                continue
        storage, form = get_form(v)
        if storage.startswith("mixed"):
            raise TypeError("pin '%s' has '%s' data" % (pin, storage))
        if storage == "pure-plain":
            if isinstance(form, str):
                vv = str(v)
                if not vv.endswith("\n"):
                    vv += "\n"
                if pin.find(".") == -1 and len(vv) <= 1000:
                    env[pin] = vv.rstrip("\n")
            else:
                vv = json.dumps(v)
            _write_file(pin, vv, "w", directory=directory)
        elif isinstance(v, bytes):
            _write_file(pin, v, "bw", directory=directory)
        else:
            if v.dtype == np.uint8 and v.ndim == 1:
                vv = v.tobytes()
                with open(job_path(pin), "bw") as pinf:
                    pinf.write(vv)
            else:
                with open(job_path(pin), "bw") as pinf:
                    np.save(pinf, v, allow_pickle=False)

    bash_header = """set -u -e
trap 'jobs -p | xargs -r kill' EXIT
"""
    if conda_environment_:
        conda_root = os.environ.get("CONDA_ROOT")
        if not conda_root:
            try:
                conda_root = (
                    subprocess.check_output(
                        ["conda", "info", "--base"],
                        stderr=subprocess.DEVNULL,
                        cwd=directory,
                    )
                    .decode()
                    .strip()
                )
            except Exception:
                pass
        if not conda_root:
            raise RuntimeError(
                "Cannot activate conda environment: "
                "could not determine conda root directory"
            )
        bash_header += f"""
source {conda_root}/etc/profile.d/conda.sh
conda activate {conda_environment_}
"""

    bashcode2 = bash_header + bashcode

    script_lines = ["#!/bin/bash\n", ": > .seamless-started\n"]
    for key in sorted(set(pins_)):
        if key in ("pins_", "bashcode") or key.startswith(META_FILE_PREFIX):
            continue
        if key not in env or not _is_shell_env_name(key):
            continue
        script_lines.append(f"export {key}={shlex.quote(str(env[key]))}\n")
    script_lines.append(bashcode2)
    script_path = job_path(script_name)
    with open(script_path, "w", encoding="utf-8") as script_file:
        script_file.writelines(script_lines)
    os.chmod(script_path, 0o755)

    return env, bashcode2


def execute_bash(bashcode, pins_, conda_environment_, PINS, FILESYSTEM, OUTPUTPIN):
    from . import SeamlessStreamTransformationError

    env = os.environ.copy()

    def read_data(data):
        if OUTPUTPIN in ("bytes", "folder", "deepfolder"):
            return data
        try:
            npdata = BytesIO(data)
            return np.load(npdata)
        except (ValueError, OSError):
            try:
                sdata = data.decode()
            except Exception:
                return np.frombuffer(data, dtype=np.uint8)
            try:
                return json.loads(sdata)
            except ValueError:
                return sdata

    process = None
    tempdir = None
    notify_path = None
    notify_fd = None
    selector = None
    output_streams = []

    def _kill_process_group():
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def sighandler(signalnum, frame):
        _kill_process_group()
        if tempdir is not None:
            shutil.rmtree(tempdir, ignore_errors=True)
        raise SystemExit()

    try:
        tempdir = os.path.abspath(
            tempfile.mkdtemp(prefix="seamless-bash-transformer")
        )
        resultfile = os.path.join(tempdir, "RESULT")
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, sighandler)
        write_bash_job(
            bashcode,
            pins_,
            conda_environment_,
            PINS,
            FILESYSTEM,
            directory=tempdir,
        )
        child_env = os.environ.copy()
        holder = current_holder()
        notify_holder = holder if holder is not None and holder.held else None
        if notify_holder is not None:
            candidate_notify_path = tempdir + ".notify"
            os.mkfifo(candidate_notify_path, 0o600)
            notify_path = candidate_notify_path
            notify_fd = os.open(notify_path, os.O_RDWR | os.O_NONBLOCK)
            child_env[NOTIFY_ENV] = notify_path
        else:
            child_env.pop(NOTIFY_ENV, None)

        process = subprocess.Popen(
            ["bash", "transform.sh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            start_new_session=True,
            cwd=tempdir,
        )
        stdout = process.stdout
        stderr = process.stderr
        assert stdout is not None and stderr is not None
        selector = selectors.DefaultSelector()
        pipes = ((stdout, "stdout"), (stderr, "stderr"))
        pending = {stream: bytearray() for stream, _ in pipes}
        registered_pipes = set()
        for stream, stream_name in pipes:
            output_streams.append(stream)
            selector.register(stream, selectors.EVENT_READ, stream_name)
            registered_pipes.add(stream)
        notify_registered = notify_fd is not None
        if notify_registered:
            selector.register(notify_fd, selectors.EVENT_READ, "notify")

        def write_output(stream_name, line):
            try:
                text = line.decode()
            except UnicodeDecodeError:
                text = line.decode(errors="ignore")
            if stream_name == "stdout" and text.strip() == "":
                return
            getattr(sys, stream_name).write(text)

        def read_notify_fifo():
            if notify_fd is None:
                return
            try:
                data = os.read(notify_fd, 65536)
            except OSError:
                return
            if data and notify_holder is not None:
                notify_holder.drop()

        while process.poll() is None or registered_pipes or notify_registered:
            if process.poll() is not None and notify_registered:
                # Drain a signal sent just before process exit, then stop
                # watching. Descendants cannot signal after the script exits.
                read_notify_fifo()
                selector.unregister(notify_fd)
                notify_registered = False
            if process.poll() is not None and not registered_pipes:
                break
            if not selector.get_map():
                process.wait()
                break
            timeout = 0.1 if notify_registered else None
            for key, _ in selector.select(timeout):
                if key.data == "notify":
                    read_notify_fifo()
                    continue
                stream = cast(IO[bytes], key.fileobj)
                chunk = os.read(stream.fileno(), 65536)
                if not chunk:
                    selector.unregister(stream)
                    registered_pipes.discard(stream)
                    tail = pending[stream]
                    if tail:
                        write_output(key.data, bytes(tail))
                        tail.clear()
                    continue
                buffer = pending[stream]
                buffer.extend(chunk)
                while True:
                    newline = buffer.find(b"\n")
                    if newline < 0:
                        break
                    line = bytes(buffer[: newline + 1])
                    del buffer[: newline + 1]
                    write_output(key.data, line)
        process.wait()

        if not os.path.exists(os.path.join(tempdir, ".seamless-started")):
            raise SeamlessStreamTransformationError(
                "\nBash transformer exception\n"
                "==========================\n\n"
                "Error: Environment was not entered; transform.sh could not be started\n"
                f"Return code: {process.returncode}\n\n"
                "*************************************************\n"
                "* Command\n"
                "*************************************************\n"
                "bash transform.sh\n"
                "*************************************************\n"
            )

        if process.returncode:
            raise SeamlessStreamTransformationError(
                """
Bash transformer exception
==========================

Error: Return code {}

*************************************************
* Command
*************************************************
{}
*************************************************
""".format(
                    process.returncode, bashcode
                )
            )
        if not os.path.exists(resultfile):
            msg = """
Bash transformer exception
==========================

Error: Result file/folder RESULT does not exist

*************************************************
* Command
*************************************************
{}
*************************************************
""".format(
                bashcode
            )
            raise SeamlessStreamTransformationError(msg)

        result_is_dir = os.path.isdir(resultfile)
        if result_is_dir:
            result = {}
            for dirpath, _, filenames in os.walk(resultfile):
                for filename in filenames:
                    full_filename = os.path.join(dirpath, filename)
                    assert full_filename.startswith(resultfile + "/")
                    member = full_filename[len(resultfile) + 1 :]
                    data = open(full_filename, "rb").read()
                    rdata = read_data(data)
                    result[member] = rdata
            if not len(result):
                msg = """
Bash transformer exception
==========================

Error: Result folder RESULT is empty

*************************************************
* Command
*************************************************
{}
*************************************************
""".format(
                    bashcode
                )
                raise SeamlessStreamTransformationError(msg)

        else:
            with open(resultfile, "rb") as f:
                resultdata = f.read()
            result = read_data(resultdata)
    finally:
        _kill_process_group()
        if selector is not None:
            try:
                selector.close()
            except Exception:
                pass
        for stream in output_streams:
            try:
                stream.close()
            except Exception:
                pass
        if notify_fd is not None:
            try:
                os.close(notify_fd)
            except OSError:
                pass
        if notify_path is not None:
            try:
                os.unlink(notify_path)
            except OSError:
                pass
        if tempdir is not None:
            shutil.rmtree(tempdir, ignore_errors=True)

    return result


__all__ = ["execute_bash", "write_bash_job"]
