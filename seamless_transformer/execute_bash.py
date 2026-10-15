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
import uuid
from io import BytesIO
from typing import IO, cast

import numpy as np

from seamless import Buffer, Checksum, CacheMissError

from seamless.util.mixed.get_form import get_form
from .compression_utils import (
    compress_bytes,
    decompress_bytes,
    strip_compression_suffix,
)
from .process_lock import NOTIFY_ENV, current_holder
from .bash_cancellation import register_cleanup
from .container import (
    build_container_command,
    remove_container,
    select_container_runtime,
    validate_container_spec,
)


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


def _conda_run_command(conda_environment):
    """Build the conda-run argv for a named environment or prefix."""
    executable = os.environ.get("SEAMLESS_CONDA_EXE") or os.environ.get("CONDA_EXE")
    if executable is None:
        conda_root = os.environ.get("CONDA_ROOT")
        executable = (
            os.path.join(conda_root, "bin", "conda") if conda_root else "conda"
        )
    executable = os.path.expanduser(executable)
    resolved_executable = shutil.which(executable)
    if resolved_executable is None:
        raise RuntimeError(
            "Cannot find conda launcher {!r}; set SEAMLESS_CONDA_EXE or ensure "
            "conda is on PATH".format(executable)
        )
    resolved_executable = os.path.abspath(resolved_executable)

    environment_flag = "-p" if "/" in conda_environment else "-n"
    command = [resolved_executable, "run", environment_flag, conda_environment]
    if os.path.basename(resolved_executable) not in ("mamba", "micromamba"):
        command.append("--no-capture-output")
    command.extend(["bash", "transform-inner.sh"])
    return command


def _which_check_lines(which_):
    if not which_:
        return []
    lines = ["if ! (\n", "  missing_binaries=0\n", "  for binary in"]
    lines.extend(" " + shlex.quote(str(binary)) for binary in which_)
    lines.extend(
        [
            "; do\n",
            '    if ! command -v "$binary" >/dev/null 2>&1; then\n',
            '      printf \'%s\\n\' "$binary" >> .seamless-missing\n',
            "      missing_binaries=1\n",
            "    fi\n",
            "  done\n",
            '  [ "$missing_binaries" -eq 0 ]\n',
            "); then\n",
            "  exit 1\n",
            "fi\n",
        ]
    )
    return lines


def _container_can_read(path, read_directories):
    """Return whether a buffer path resolves under a directory we bind."""
    resolved = os.path.realpath(path)
    for directory in read_directories or ():
        root = os.path.realpath(directory)
        try:
            if os.path.commonpath((root, resolved)) == root:
                return True
        except ValueError:
            pass
    return False


def _safe_directory_member_path(root, key):
    if not isinstance(key, str) or not key:
        raise ValueError(f"Invalid directory member name: {key!r}")
    if os.path.isabs(key) or ".." in key.replace("\\", "/").split("/"):
        raise ValueError(f"Invalid directory member name: {key!r}")
    path = os.path.abspath(os.path.join(root, key))
    if os.path.commonpath((os.path.abspath(root), path)) != os.path.abspath(root):
        raise ValueError(f"Invalid directory member name: {key!r}")
    return path


def write_directory_pin(
    pin_path,
    structure,
    celltype,
    *,
    read_directories=None,
    container=False,
    members_are_checksums=None,
    compression_suffix=None,
):
    """Write a folder pin using the bash job's materialization rules."""
    if celltype == "deepfolder":
        raise TypeError(
            "Bash transformer pins cannot be deepfolder; declare it folder or plain"
        )
    if celltype != "folder":
        raise ValueError(f"Unsupported directory pin celltype {celltype!r}")
    if not isinstance(structure, dict):
        raise TypeError(f"{celltype} pin must contain a mapping of members")

    from .transformation_namespace import _find_filesystem_path

    root = os.path.abspath(pin_path)
    os.makedirs(root, exist_ok=True)
    if members_are_checksums is None:
        members_are_checksums = False
    if compression_suffix is None:
        _pin_base, target_suffix = strip_compression_suffix(os.path.basename(pin_path))
    else:
        target_suffix = compression_suffix

    for key, value in structure.items():
        member_is_checksum = isinstance(value, Checksum) or (
            members_are_checksums and isinstance(value, str)
        )
        if member_is_checksum:
            checksum = Checksum(value)
            source_path, source_suffix = _find_filesystem_path(
                checksum, "file", read_directories
            )
            path_is_reachable = (
                source_path is not None
                and (not container or _container_can_read(source_path, read_directories))
            )
            member_path = _safe_directory_member_path(
                root, key + (target_suffix or "")
            )
            os.makedirs(os.path.dirname(member_path), exist_ok=True)
            if path_is_reachable and source_suffix == target_suffix:
                os.symlink(os.path.abspath(source_path), member_path)
                continue
            if source_path is not None:
                with open(source_path, "rb") as handle:
                    data = handle.read()
                if source_suffix != target_suffix:
                    if source_suffix is not None:
                        data = decompress_bytes(data, source_suffix)
                    if target_suffix is not None:
                        data = compress_bytes(data, target_suffix)
            else:
                buffer = checksum.resolve()
                if buffer is None:
                    raise CacheMissError(
                        f"folder member {key!r} ({checksum.hex()}) could not be resolved"
                    )
                data = buffer.content
                if target_suffix is not None:
                    data = compress_bytes(data, target_suffix)
            with open(member_path, "wb") as handle:
                handle.write(data)
            continue

        if isinstance(value, Buffer):
            data = value.content
        elif isinstance(value, bytes):
            data = value
        else:
            data = Buffer(value, "text" if isinstance(value, str) else "mixed").content
        member_path = _safe_directory_member_path(root, key + (target_suffix or ""))
        os.makedirs(os.path.dirname(member_path), exist_ok=True)
        if target_suffix is not None:
            data = compress_bytes(data, target_suffix)
        with open(member_path, "wb") as handle:
            handle.write(data)


def write_bash_job(
    bashcode,
    pins_,
    conda_environment_,
    PINS,
    FILESYSTEM,
    *,
    script_name="transform.sh",
    directory=None,
    which_=(),
    conda_command_=None,
    docker_=None,
    container_runtime_=None,
    tf_checksum_=None,
    process_id_=None,
    container_name_=None,
    read_directories_=None,
    portable_job_directory_=True,
    container_command_=None,
):
    """Materialize a bash transformation job in the supplied directory."""

    def job_path(path):
        if directory is None:
            return path
        return os.path.join(directory, path)

    if conda_environment_ and conda_command_ is None:
        conda_command_ = _conda_run_command(conda_environment_)

    if read_directories_ is None:
        from .transformation_namespace import _get_read_folder_directories

        read_directories_ = _get_read_folder_directories()

    container_command = container_command_
    if docker_ is not None:
        docker_ = validate_container_spec(docker_)
        if conda_environment_:
            raise ValueError(
                "docker and conda_environment cannot be used together; pick one, "
                "or activate the conda environment in the script itself"
            )
        if container_command is None:
            if container_runtime_ is None:
                container_runtime_ = select_container_runtime(docker_["name"])
            job_directory = os.getcwd() if directory is None else directory
            container_command = build_container_command(
                container_runtime_,
                docker_,
                job_directory,
                read_directories_,
                tf_checksum=tf_checksum_,
                process_id=process_id_,
                container_name=container_name_,
                portable_job_directory=portable_job_directory_,
            )

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
                can_link = not container_command or _container_can_read(
                    v, read_directories_
                )
                if can_link and found_suffix == pin_suffix:
                    os.symlink(v, pin_path)
                else:
                    with open(job_path(v), "rb") as handle:
                        data = handle.read()
                    if found_suffix != pin_suffix:
                        if found_suffix is not None:
                            data = decompress_bytes(data, found_suffix)
                        if pin_suffix is not None:
                            data = compress_bytes(data, pin_suffix)
                    _write_file(pin, data, "bw", directory=directory)
                continue
            elif FILESYSTEM[pin]["mode"] == "directory":
                fs_entry = FILESYSTEM[pin]
                directory_pin, compression_suffix = strip_compression_suffix(pin)
                celltype = fs_entry.get("celltype", "folder")
                write_directory_pin(
                    job_path(directory_pin),
                    v,
                    celltype,
                    read_directories=read_directories_,
                    container=container_command is not None,
                    members_are_checksums=fs_entry.get(
                        "members_are_checksums", False
                    ),
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

    bash_header = "set -u -e\n"
    if container_command is None:
        bash_header += "trap 'jobs -p | xargs -r kill' EXIT\n"
    bashcode2 = bash_header + bashcode

    script_lines = ["#!/bin/bash\n"]
    for key in sorted(set(pins_)):
        if key in ("pins_", "bashcode") or key.startswith(META_FILE_PREFIX):
            continue
        if key not in env or not _is_shell_env_name(key):
            continue
        script_lines.append(f"export {key}={shlex.quote(str(env[key]))}\n")
    script_lines.append(": > .seamless-started\n")
    script_lines.extend(_which_check_lines(which_))
    script_lines.append(bashcode2)

    if conda_command_ is not None or container_command is not None:
        inner_script_path = job_path("transform-inner.sh")
        with open(inner_script_path, "w", encoding="utf-8") as script_file:
            script_file.writelines(script_lines)
        os.chmod(inner_script_path, 0o755)

        if container_command is not None:
            launcher_command = shlex.join(container_command)
            portable_mount = shlex.quote(
                "type=bind,src=__SEAMLESS_JOB_DIR__,dst=/seamless-job"
            )
            launcher_command = launcher_command.replace(
                portable_mount,
                'type=bind,src="$PWD",dst=/seamless-job',
            )
            portable_job_bind = shlex.quote(
                "__SEAMLESS_JOB_DIR__:/seamless-job"
            )
            launcher_command = launcher_command.replace(
                portable_job_bind, '"$PWD":/seamless-job'
            )
        else:
            launcher_command = shlex.join(conda_command_)
        launcher_lines = ["#!/bin/bash\n", "exec " + launcher_command + "\n"]
    else:
        launcher_lines = script_lines

    script_path = job_path(script_name)
    with open(script_path, "w", encoding="utf-8") as script_file:
        script_file.writelines(launcher_lines)
    os.chmod(script_path, 0o755)

    return env, bashcode2


def execute_bash(
    bashcode,
    pins_,
    conda_environment_,
    PINS,
    FILESYSTEM,
    OUTPUTPIN,
    *,
    which_=(),
    docker_=None,
    container_runtime_=None,
    tf_checksum_=None,
    should_abort_=None,
):
    from . import SeamlessStreamTransformationError
    from seamless.error_envelope import ExecutionCanceledError

    env = os.environ.copy()

    def check_cancelled():
        if should_abort_ is not None and should_abort_():
            raise ExecutionCanceledError("Transformation was canceled")

    check_cancelled()

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

    docker_spec = None
    if docker_ is not None:
        docker_spec = validate_container_spec(docker_)
        if conda_environment_:
            raise ValueError(
                "docker and conda_environment cannot be used together; pick one, "
                "or activate the conda environment in the script itself"
            )
        if container_runtime_ is None:
            container_runtime_ = select_container_runtime(docker_spec["name"])
    container_name = (
        "seamless-{}".format(uuid.uuid4().hex[:12]) if docker_spec is not None else None
    )
    process = None
    process_started = threading.Event()
    tempdir = None
    notify_path = None
    notify_fd = None
    selector = None
    output_streams = []

    def _kill_process_group():
        if process is None:
            # SIGTERM can arrive while the thread calling Popen is still
            # returning the new process object. Give it a short chance to publish
            # the process handle before the worker exits.
            process_started.wait(0.1)
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def sighandler(signalnum, frame):
        _kill_process_group()
        if container_name is not None:
            remove_container(container_runtime_, container_name, detached=True)
        if tempdir is not None:
            shutil.rmtree(tempdir, ignore_errors=True)
        raise SystemExit()

    def cleanup_launch():
        _kill_process_group()
        if container_name is not None:
            remove_container(container_runtime_, container_name, detached=True)

    unregister_cleanup = register_cleanup(tf_checksum_, cleanup_launch)

    try:
        check_cancelled()
        tempdir = os.path.abspath(
            tempfile.mkdtemp(prefix="seamless-bash-transformer")
        )
        resultfile = os.path.join(tempdir, "RESULT")
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, sighandler)
        conda_command = (
            _conda_run_command(conda_environment_) if conda_environment_ else None
        )
        container_command = None
        read_directories = None
        if docker_spec is not None:
            from .transformation_namespace import _get_read_folder_directories

            read_directories = _get_read_folder_directories()
            if hasattr(tf_checksum_, "hex"):
                tf_checksum_ = tf_checksum_.hex()
            container_command = build_container_command(
                container_runtime_,
                docker_spec,
                tempdir,
                read_directories,
                tf_checksum=tf_checksum_,
                process_id=os.getpid(),
                container_name=container_name,
            )
        write_bash_job(
            bashcode,
            pins_,
            conda_environment_,
            PINS,
            FILESYSTEM,
            directory=tempdir,
            which_=which_,
            conda_command_=conda_command,
            docker_=docker_spec,
            container_runtime_=container_runtime_,
            tf_checksum_=tf_checksum_,
            process_id_=os.getpid(),
            container_name_=container_name,
            read_directories_=read_directories,
            container_command_=container_command,
        )
        child_env = os.environ.copy()
        holder = current_holder()
        notify_holder = (
            holder
            if docker_spec is None and holder is not None and holder.held
            else None
        )
        if notify_holder is not None:
            candidate_notify_path = tempdir + ".notify"
            os.mkfifo(candidate_notify_path, 0o600)
            notify_path = candidate_notify_path
            notify_fd = os.open(notify_path, os.O_RDWR | os.O_NONBLOCK)
            child_env[NOTIFY_ENV] = notify_path
        else:
            child_env.pop(NOTIFY_ENV, None)

        try:
            process = subprocess.Popen(
                ["bash", "transform.sh"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=child_env,
                start_new_session=True,
                cwd=tempdir,
            )
        finally:
            process_started.set()
        check_cancelled()
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
            stream = getattr(sys, stream_name)
            stream.write(text)
            stream.flush()

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
            failure_hint = ""
            if docker_spec is not None:
                environment_description = "{} image {!r}".format(
                    container_runtime_, docker_spec["name"]
                )
                launcher_line = "exec " + shlex.join(container_command)
                if process.returncode == 125:
                    failure_hint = (
                        "Hint: the container runtime could not start the image; "
                        "check that the image is present and that every bind-mount "
                        "source exists"
                    )
                elif process.returncode in (126, 127):
                    failure_hint = "Hint: bash was not found in the container image"
            elif conda_command is not None:
                environment_description = "Conda environment {!r}".format(
                    conda_environment_
                )
                launcher_line = "exec " + shlex.join(conda_command)
            else:
                environment_description = "Environment"
                launcher_line = "bash transform.sh"
            failure_message = (
                "\nBash transformer exception\n"
                "==========================\n\n"
                f"Error: {environment_description} was not entered; transform.sh could not be started\n"
                f"Return code: {process.returncode}\n"
            )
            if failure_hint:
                failure_message += f"{failure_hint}\n"
            failure_message += (
                f"Launcher: {launcher_line}\n\n"
                "*************************************************\n"
                "* Command\n"
                "*************************************************\n"
                f"{launcher_line}\n"
                "*************************************************\n"
            )
            raise SeamlessStreamTransformationError(failure_message)

        missing_marker = os.path.join(tempdir, ".seamless-missing")
        if os.path.exists(missing_marker):
            with open(missing_marker, encoding="utf-8") as missing_file:
                missing_binaries = [line.rstrip("\n") for line in missing_file]
            checked_where = (
                "inside conda environment {!r}".format(conda_environment_)
                if conda_environment_
                else "inside {} image {!r}".format(
                    container_runtime_, docker_spec["name"]
                )
                if docker_spec is not None
                else "in the bash script environment"
            )
            raise SeamlessStreamTransformationError(
                "Required environment binaries are unavailable: {}\n"
                "Checked {}".format(", ".join(missing_binaries), checked_where)
            )

        if process.returncode:
            environment_line = (
                "Environment: conda environment {!r}".format(conda_environment_)
                if conda_environment_
                else "Environment: {} image {!r}".format(
                    container_runtime_, docker_spec["name"]
                )
                if docker_spec is not None
                else ""
            )
            if docker_spec is not None and process.returncode == 137:
                environment_line += (
                    "\nHint: the container job was killed, possibly for exceeding "
                    "a memory limit"
                )
            raise SeamlessStreamTransformationError(
                """
Bash transformer exception
==========================

Error: Return code {}
{}

*************************************************
* Command
*************************************************
{}
*************************************************
""".format(
                    process.returncode, environment_line, bashcode
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
        if OUTPUTPIN in ("folder", "deepcell") and not result_is_dir:
            raise SeamlessStreamTransformationError(
                f"Bash {OUTPUTPIN} result requires RESULT to be a directory"
            )
        if result_is_dir:
            result = {}
            for dirpath, _, filenames in os.walk(resultfile):
                for filename in filenames:
                    full_filename = os.path.join(dirpath, filename)
                    assert full_filename.startswith(resultfile + "/")
                    member = full_filename[len(resultfile) + 1 :]
                    data = open(full_filename, "rb").read()
                    if OUTPUTPIN == "deepcell":
                        try:
                            Buffer(data).get_value("mixed")
                        except Exception as exc:
                            raise SeamlessStreamTransformationError(
                                f"Bash deepcell result member {member!r} "
                                "is not a valid mixed buffer"
                            ) from exc
                        rdata = data
                    else:
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
        try:
            cleanup_launch()
            if container_name is not None:
                remove_container(container_runtime_, container_name)
        finally:
            unregister_cleanup()
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


__all__ = ["execute_bash", "write_bash_job", "write_directory_pin"]
