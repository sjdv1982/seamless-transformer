import os
import re
import subprocess

from seamless import Buffer, Checksum

from seamless_transformer.api import run_transformation

_CHECKSUM_RE = re.compile(r"\b[a-f0-9]{64}\b")
_HELLO_CHECKSUM = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


def _run_command(args, *, cwd, env=None):
    run_env = os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    if env:
        run_env.update(env)
    return subprocess.run(
        args,
        cwd=cwd,
        env=run_env,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_success(proc):
    assert proc.returncode == 0, "\n".join(
        line
        for line in (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
        if line
    )


def _prepare_uploaded_echo_job(tmp_path):
    workdir = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    job_dir = tmp_path / "job"
    workdir.mkdir()
    cache_dir.mkdir()
    env = {"SEAMLESS_CACHE": str(cache_dir)}

    proc = _run_command(
        [
            "seamless-run",
            "--dry-run",
            "--upload",
            "-q",
            "-j",
            str(job_dir),
            "-c",
            "echo hello",
        ],
        cwd=workdir,
        env=env,
    )
    _assert_success(proc)
    checksums = _CHECKSUM_RE.findall(proc.stdout)
    assert checksums, proc.stdout
    checksum = checksums[-1]
    checksum_file = job_dir / "transformation.json.CHECKSUM"
    assert checksum_file.read_text().strip() == checksum
    return workdir, env, checksum, checksum_file


def _write_daskserver_config(workdir):
    (workdir / "seamless.yaml").write_text(
        "- project: cmd-test\n- execution: remote\n- remote: daskserver\n"
    )
    (workdir / "seamless.profile.yaml").write_text("- cluster: local\n")


def _extract_labeled_checksum(label, proc):
    match = re.search(
        rf"^{label}\s+([a-f0-9]{{64}})$",
        (proc.stdout or "") + "\n" + (proc.stderr or ""),
        re.MULTILINE,
    )
    assert match, "\n".join(
        line
        for line in (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
        if line
    )
    return match.group(1)


def test_run_transformation_cli_wires_strict(monkeypatch, capsys):
    checksum = Checksum("1" * 64)
    calls = []
    fake_transformation = object()

    monkeypatch.setattr(run_transformation, "_parse_checksum", lambda _arg: checksum)
    monkeypatch.setattr(
        run_transformation,
        "_resolve_transformation_dict",
        lambda _checksum: {"__output__": ("result", "mixed", None)},
    )

    def fake_transformation_from_dict(
        transformation_dict,
        *,
        tf_dunder,
        scratch,
        strict_dunder=False,
    ):
        calls.append(
            {
                "transformation_dict": transformation_dict,
                "tf_dunder": tf_dunder,
                "scratch": scratch,
                "strict_dunder": strict_dunder,
            }
        )
        return fake_transformation

    def fake_compute_transformation_sync(transformation, *, require_value):
        assert transformation is fake_transformation
        calls[-1]["require_value"] = require_value
        return Checksum("2" * 64)

    monkeypatch.setattr(
        run_transformation,
        "transformation_from_dict",
        fake_transformation_from_dict,
    )
    monkeypatch.setattr(
        run_transformation,
        "compute_transformation_sync",
        fake_compute_transformation_sync,
    )

    assert run_transformation._main(["--strict", checksum.hex()]) == 0
    assert calls[0]["strict_dunder"] is True
    assert calls[0]["scratch"] is False
    assert calls[0]["require_value"] is False
    assert capsys.readouterr().out.strip() == "2" * 64


def test_run_transformation_cli_loads_sibling_dunder(monkeypatch, tmp_path, capsys):
    checksum = "1" * 64
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    checksum_file = job_dir / "transformation.json.CHECKSUM"
    checksum_file.write_text(checksum)
    (job_dir / "dunder.json").write_text('{"__meta__": {"direct_print": true}}')
    calls = []
    fake_transformation = object()

    monkeypatch.setattr(
        run_transformation,
        "_resolve_transformation_dict",
        lambda _checksum: {
            "__language__": "bash",
            "__output__": ("result", "mixed", None),
            "code": ("bytes", None, "2" * 64),
        },
    )

    def fake_transformation_from_dict(
        transformation_dict,
        *,
        tf_dunder,
        scratch,
        strict_dunder=False,
    ):
        calls.append(
            {
                "transformation_dict": transformation_dict,
                "tf_dunder": tf_dunder,
                "scratch": scratch,
                "strict_dunder": strict_dunder,
            }
        )
        return fake_transformation

    def fake_compute_transformation_sync(transformation, *, require_value):
        assert transformation is fake_transformation
        return Checksum("3" * 64)

    monkeypatch.setattr(
        run_transformation,
        "transformation_from_dict",
        fake_transformation_from_dict,
    )
    monkeypatch.setattr(
        run_transformation,
        "compute_transformation_sync",
        fake_compute_transformation_sync,
    )

    assert run_transformation._main([str(checksum_file)]) == 0
    assert calls[0]["tf_dunder"]["__meta__"] == {"direct_print": True}
    assert capsys.readouterr().out.strip() == "3" * 64


def test_run_transformation_cli_replays_uploaded_checksum(tmp_path):
    workdir, env, checksum, _checksum_file = _prepare_uploaded_echo_job(tmp_path)

    proc = _run_command(
        [
            "seamless-run-transformation",
            checksum,
        ],
        cwd=workdir,
        env=env,
    )

    _assert_success(proc)
    assert proc.stdout.splitlines()[-1].strip() == _HELLO_CHECKSUM


def test_run_transformation_cli_replays_uploaded_job_checksum_file(tmp_path):
    workdir, env, _checksum, checksum_file = _prepare_uploaded_echo_job(tmp_path)

    proc = _run_command(
        [
            "seamless-run-transformation",
            str(checksum_file),
        ],
        cwd=workdir,
        env=env,
    )

    _assert_success(proc)
    assert proc.stdout.splitlines()[-1].strip() == _HELLO_CHECKSUM


def test_run_transformation_cli_replays_uploaded_job_checksum_file_with_daskserver_config(
    tmp_path,
):
    workdir, env, _checksum, checksum_file = _prepare_uploaded_echo_job(tmp_path)
    _write_daskserver_config(workdir)

    proc = _run_command(
        [
            "seamless-run-transformation",
            str(checksum_file),
        ],
        cwd=workdir,
        env=env,
    )

    _assert_success(proc)
    assert proc.stdout.splitlines()[-1].strip() == _HELLO_CHECKSUM


def test_run_transformation_cli_replays_delayed_python_transformation_checksum(
    tmp_path,
):
    workdir = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    workdir.mkdir()
    cache_dir.mkdir()
    env = {"SEAMLESS_CACHE": str(cache_dir)}
    producer = workdir / "make_delayed_checksum.py"
    producer.write_text(
        "\n".join(
            [
                "import seamless",
                "import seamless.config as seamless_config",
                "from seamless.transformer import delayed",
                "",
                "seamless_config.init()",
                "",
                "@delayed",
                "def add(a, b):",
                "    return a + b",
                "",
                "tf = add(19, 23)",
                "tf.construct()",
                "transformation_buffer = tf.transformation_checksum.resolve()",
                "transformation_buffer.incref()",
                'print("TF_CHECKSUM", tf.transformation_checksum.hex())',
                "seamless.close()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = _run_command(["python", str(producer)], cwd=workdir, env=env)
    _assert_success(proc)
    checksum = _extract_labeled_checksum("TF_CHECKSUM", proc)

    proc = _run_command(
        [
            "seamless-run-transformation",
            checksum,
        ],
        cwd=workdir,
        env=env,
    )

    _assert_success(proc)
    expected = Buffer(42, "mixed").get_checksum().hex()
    assert proc.stdout.splitlines()[-1].strip() == expected
