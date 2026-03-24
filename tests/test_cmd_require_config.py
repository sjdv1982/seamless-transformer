import sys

import pytest

from seamless_transformer.cmd.api import main as cmd_main


def test_require_config_file_accepts_profile_yaml(tmp_path):
    (tmp_path / "seamless.profile.yaml").write_text("- execution: process\n")
    cmd_main._require_config_file(str(tmp_path))


def test_main_requires_config_before_stage(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    stage_calls = []

    def fake_set_stage(*args, **kwargs):
        stage_calls.append((args, kwargs))
        raise AssertionError("set_stage should not be called before config detection")

    def fake_err(*args):
        raise RuntimeError(" ".join(str(arg) for arg in args))

    monkeypatch.setattr(cmd_main.seamless.config, "set_stage", fake_set_stage)
    monkeypatch.setattr(cmd_main, "err", fake_err)
    monkeypatch.setattr(sys, "argv", ["seamless-run", "--stage", "demo", "echo"])

    with pytest.raises(RuntimeError, match="seamless.yaml or seamless.profile.yaml"):
        cmd_main._main()

    assert not stage_calls


def test_main_local_bypasses_config_requirement(monkeypatch):
    called = False

    def fake_require_config_file(workdir):
        nonlocal called
        called = True
        raise AssertionError("config detection should be skipped in --local mode")

    monkeypatch.setattr(cmd_main, "_require_config_file", fake_require_config_file)
    monkeypatch.setattr(sys, "argv", ["seamless-run", "--local"])

    assert cmd_main._main() == 1
    assert called is False


def test_main_seamless_local_env_bypasses_config_requirement(monkeypatch):
    called = False

    def fake_require_config_file(workdir):
        nonlocal called
        called = True
        raise AssertionError(
            "config detection should be skipped when SEAMLESS_LOCAL is set"
        )

    monkeypatch.setenv("SEAMLESS_LOCAL", "1")
    monkeypatch.setattr(cmd_main, "_require_config_file", fake_require_config_file)
    monkeypatch.setattr(sys, "argv", ["seamless-run"])

    assert cmd_main._main() == 1
    assert called is False
