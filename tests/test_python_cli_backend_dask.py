import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def python_cli_dask_cluster(tmp_path_factory):
    harness_path = Path(__file__).parent / "cancellation" / "_harness.py"
    spec = importlib.util.spec_from_file_location(
        "python_cli_dask_harness", harness_path
    )
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    context = harness.build_cluster(
        tmp_path_factory.mktemp("python_cli_dask"),
        remote_kind="daskserver",
        cluster_name=f"python-cli-dask-{os.getpid()}",
        port_base=63000,
        walltime="1h",
        seamless_workers=2,
        dask_resources_s=2,
    )
    try:
        cluster = context.__enter__()
    except PermissionError as exc:
        pytest.skip(f"local daskserver harness is unavailable due to permissions: {exc}")
    try:
        yield cluster
    except BaseException as exc:
        context.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        context.__exit__(None, None, None)


def _run_remote_script(cluster, source, *, timeout=240):
    script_path = cluster.workdir / f"python-cli-{uuid.uuid4().hex}.py"
    script_path.write_text(source, encoding="utf-8")
    try:
        result = cluster.run_command([sys.executable, str(script_path)], timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        def tail(value):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            return (value or "")[-8000:]

        raise AssertionError(
            f"remote Python script timed out after {timeout}s; "
            f"stdout tail: {tail(exc.stdout)!r}; stderr tail: {tail(exc.stderr)!r}"
        ) from exc
    finally:
        script_path.unlink(missing_ok=True)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = "PYTHON_CLI_DASK_RESULT="
    lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
    assert lines, result.stdout + result.stderr
    return json.loads(lines[-1][len(marker) :])


@pytest.mark.parametrize("backend", ["mem", "cli"])
@pytest.mark.parametrize("driver", [False, True], ids=["body", "driver"])
def test_nested_python_cli_submission_runs_on_cluster_and_reuses_cache(
    python_cli_dask_cluster, driver, backend
):
    first_marker = uuid.uuid4().hex
    second_marker = uuid.uuid4().hex
    source = f"""
import json
import os
import seamless.config as seamless_config
from seamless.transformer import Transformer

seamless_config.init()
driver = {driver!r}
body = '''
import os
import uuid
from seamless.transformer import direct

@direct
def child():
    import os
    import uuid
    return (os.getpid(), uuid.uuid4().hex)

first = child()
second = child()
result = (parent_marker, os.getpid(), first, second)
'''

def run_parent(marker):
    tf = Transformer("python")
    tf.code = body
    tf.pins["parent_marker"] = marker
    tf.metavars.BACKEND = {backend!r}
    tf.driver = driver
    built = tf.build()
    return built.run()

first = run_parent({first_marker!r})
second = run_parent({second_marker!r})
print("PYTHON_CLI_DASK_RESULT=" + json.dumps([first, second]), flush=True)
"""
    first, second = _run_remote_script(python_cli_dask_cluster, source)

    assert first[0] == first_marker
    assert second[0] == second_marker
    assert first[0] != second[0]
    assert first[2] == first[3]
    assert second[2] == second[3]
    # The changing UUID proves the second parent reused the recorded child
    # result instead of running the same child transformation again.
    assert first[2] == second[2]
    assert first[1] != os.getpid()
    assert first[2][0] != os.getpid()


def test_python_cli_deepcell_member_resolves_on_dask(python_cli_dask_cluster):
    value = "python-cli-dask-deepcell-" + uuid.uuid4().hex
    source = f"""
import asyncio
import json
import seamless.config as seamless_config
from seamless import Buffer
from seamless.transformer import Transformer
from seamless_remote import buffer_remote

seamless_config.init()
value = {value!r}
member = Buffer(value, "text")
checksum = member.get_checksum()
assert asyncio.run(buffer_remote.write_buffer(checksum, member))

tf = Transformer("python")
tf.code = "result = values['member'].resolve().get_value('text')"
tf.celltypes.values = "deepcell"
tf.pins["values"] = {{"member": checksum}}
checksums = []
for backend in ("mem", "cli"):
    tf.metavars.BACKEND = backend
    tf.pins["parity_nonce"] = backend + {uuid.uuid4().hex!r}
    built = tf.build()
    assert built.run() == value
    checksums.append(built.result_checksum.hex())
assert checksums[0] == checksums[1]
print("PYTHON_CLI_DASK_RESULT=" + json.dumps(value), flush=True)
"""
    assert _run_remote_script(python_cli_dask_cluster, source) == value


@pytest.mark.parametrize("driver", [False, True], ids=["body", "driver"])
def test_nested_python_backends_checksum_parity(python_cli_dask_cluster, driver):
    # The Phase 4 cache test returns UUIDs and process IDs. This companion
    # compares a deterministic nested result under the same cluster.
    source = f"""
import json
import seamless.config as seamless_config
from seamless.transformer import Transformer
seamless_config.init()
checksums = []
for backend in ("mem", "cli"):
    tf = Transformer("python")
    tf.code = '''
from seamless.transformer import direct
@direct
def child(value):
    return value + 1
result = child(41)
'''
    tf.pins["parity_nonce"] = backend + {uuid.uuid4().hex!r}
    tf.metavars.BACKEND = backend
    tf.driver = {driver!r}
    built = tf.build()
    assert built.run() == 42
    checksums.append(built.result_checksum.hex())
assert checksums[0] == checksums[1]
print("PYTHON_CLI_DASK_RESULT=" + json.dumps(checksums), flush=True)
"""
    checksums = _run_remote_script(python_cli_dask_cluster, source)
    assert checksums[0] == checksums[1]
