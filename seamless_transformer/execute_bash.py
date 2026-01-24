import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import sys
from io import BytesIO

import numpy as np

from seamless import Buffer

from seamless.util.mixed.get_form import get_form
from seamless.util.mount_directory import write_to_directory


def _write_file(pinname, data, filemode):
    if pinname.startswith("/"):
        raise ValueError("Pin {}: Absolute path is not allowed")
    path_elements = pinname.split("/")
    if ".." in path_elements:
        raise ValueError("Pin {}: .. is not allowed")
    if len(path_elements) > 1:
        parent_dir = os.path.dirname(pinname)
        os.makedirs(parent_dir, exist_ok=True)
    with open(pinname, filemode) as pinf:
        pinf.write(data)


def execute_bash(bashcode, pins_, conda_environment_, PINS, FILESYSTEM, OUTPUTPIN):
    from . import SeamlessStreamTransformationError

    from . import global_lock

    env = os.environ.copy()
    resultfile = "RESULT"

    def read_data(data):
        if OUTPUTPIN in ("bytes", "folder", "deepfolder"):
            return data
        try:
            npdata = BytesIO(data)
            return np.load(npdata)
        except (ValueError, OSError):
            try:
                try:
                    sdata = data.decode()
                except Exception:
                    return np.frombuffer(data, dtype=np.uint8)
                return json.loads(sdata)
            except ValueError:
                return sdata

    process = None
    tempdir = tempfile.mkdtemp(prefix="seamless-bash-transformer")

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
        os.chdir(old_cwd)
        shutil.rmtree(tempdir, ignore_errors=True)
        raise SystemExit()

    try:
        global_lock.acquire()
        old_cwd = os.getcwd()
        os.chdir(tempdir)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, sighandler)
        for pin in pins_:
            if pin == "pins_":
                continue
            if pin == "bashcode":
                continue
            v = PINS[pin]
            if isinstance(v, Buffer):
                v = v.content
            if pin in FILESYSTEM:
                if FILESYSTEM[pin]["filesystem"]:
                    env[pin] = v
                    pin_parent = os.path.dirname(pin)
                    if len(pin_parent):
                        os.makedirs(pin_parent, exist_ok=True)
                    os.symlink(v, pin)
                    continue
                elif FILESYSTEM[pin]["mode"] == "directory":
                    write_to_directory(
                        pin, v, cleanup=False, deep=False, text_only=False
                    )
                    env[pin] = pin
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
                _write_file(pin, vv, "w")
            elif isinstance(v, bytes):
                _write_file(pin, v, "bw")
            else:
                if v.dtype == np.uint8 and v.ndim == 1:
                    vv = v.tobytes()
                    with open(pin, "bw") as pinf:
                        pinf.write(vv)
                else:
                    with open(pin, "bw") as pinf:
                        np.save(pinf, v, allow_pickle=False)
        bash_header = """set -u -e
trap 'jobs -p | xargs -r kill' EXIT
"""
        if conda_environment_:
            CONDA_ROOT = os.environ.get("CONDA_ROOT", None)
            bash_header += f"""
source {CONDA_ROOT}/etc/profile.d/conda.sh
conda activate {conda_environment_}
"""

        bashcode2 = bash_header + bashcode
        process = subprocess.Popen(
            bashcode2,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            executable="/bin/bash",
            env=env,
            preexec_fn=os.setsid,
        )
        for line in process.stdout:
            try:
                text = line.decode()
            except UnicodeDecodeError:
                text = line.decode(errors="ignore")
            if text.strip() == "":
                continue
            print(text, end="")
        process.wait()

        if process.returncode:
            if process.stdout:
                sys.stdout.buffer.write(process.stdout.read())
            if process.stderr:
                sys.stderr.buffer.write(process.stderr.read())
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

        if os.path.isdir(resultfile):
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
        os.chdir(old_cwd)
        shutil.rmtree(tempdir, ignore_errors=True)
        global_lock.release()

    return result


__all__ = ["execute_bash"]
