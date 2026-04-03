from __future__ import annotations

from pathlib import Path

import pytest

from seamless_transformer import run


def test_write_remote_bash_job_expands_user(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    captured = {}

    def fake_write_bash_job(*args, **kwargs):
        captured["cwd"] = Path.cwd()

    monkeypatch.setattr(run, "write_bash_job", fake_write_bash_job)

    result = run._write_remote_bash_job("~/remote-job", "echo OK", [], "", {}, {})

    expected = home / "remote-job"
    assert result == str(expected)
    assert captured["cwd"] == expected


def test_write_remote_bash_job_permission_error_is_actionable(monkeypatch):
    def fake_makedirs(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(run.os, "makedirs", fake_makedirs)

    with pytest.raises(PermissionError, match="execution host"):
        run._write_remote_bash_job("/users/example/job", "echo OK", [], "", {}, {})
