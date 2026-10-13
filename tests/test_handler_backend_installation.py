"""Opt-in installation smoke checks; embedded interpreters start in fresh processes.

Run explicitly with SEAMLESS_TEST_BACKEND_INSTALLATION=1 after Phase 3 installs.
Ordinary test runs do not require these optional backend packages.
"""
import os
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("SEAMLESS_TEST_BACKEND_INSTALLATION") != "1",
    reason="optional backend installation smoke checks are opt-in",
)


def _run(command, *, timeout=300):
    completed = subprocess.run(
        command, text=True, capture_output=True, timeout=timeout,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, (
        f"{command[0]} exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "SEAMLESS_BACKEND_OK" in completed.stdout


def test_rscript_and_required_r_packages():
    executable = shutil.which("Rscript")
    assert executable is not None, "Phase 3 requires Rscript on PATH"
    _run([executable, "--vanilla", "-e", '''
        packages <- c("arrow", "jsonlite", "RcppCNPy")
        for (package in packages) {
            if (!requireNamespace(package, quietly=TRUE)) {
                stop(paste("Missing required R package", package))
            }
        }
        value <- jsonlite::fromJSON(jsonlite::toJSON(list(answer=42L), auto_unbox=TRUE))
        stopifnot(value$answer == 42L)
        cat("SEAMLESS_BACKEND_OK\\n")
    '''])


def test_ryp_import_and_basic_r_roundtrip_in_fresh_process():
    _run([sys.executable, "-c", '''
from ryp import r, to_r, to_py
to_r(41, "seamless_installation_input")
r("seamless_installation_output <- seamless_installation_input + 1")
value = to_py("seamless_installation_output", squeeze=True)
assert value == 42, repr(value)
print("SEAMLESS_BACKEND_OK")
'''])


def test_juliacall_import_json_project_and_basic_roundtrip_in_fresh_process():
    _run([sys.executable, "-c", '''
from juliacall import Main as jl
jl.seval("using JSON")
jl.seamless_installation_input = 41
assert int(jl.seval("seamless_installation_input + 1")) == 42
encoded = str(jl.seval('JSON.json(Dict("answer" => 42))'))
assert encoded == '{"answer":42}', encoded
jl.seamless_installation_json = encoded
assert int(jl.seval('JSON.parse(seamless_installation_json)["answer"]')) == 42
print("SEAMLESS_BACKEND_OK")
'''], timeout=600)