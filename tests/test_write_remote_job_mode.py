import sys

import pytest

from seamless_transformer.cmd.api import main as cmd_main


def test_write_remote_job_dry_run_does_not_force_process(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "seamless.yaml").write_text("- execution: remote\n")

    select_execution_calls = []

    def fake_require_config_file(workdir):
        return None

    def fake_set_remote_clients_from_env(*, include_dask):
        assert include_dask is True
        return False

    def fake_load_config_files():
        return None

    def fake_select_execution(mode):
        select_execution_calls.append(mode)

    def fake_get_selected_cluster():
        return "dummy-cluster"

    def fake_get_execution():
        return "remote"

    def fake_change_stage():
        raise RuntimeError("stop-after-config")

    monkeypatch.setattr(cmd_main, "_require_config_file", fake_require_config_file)
    monkeypatch.setattr(
        "seamless_config.extern_clients.set_remote_clients_from_env",
        fake_set_remote_clients_from_env,
    )
    monkeypatch.setattr(
        "seamless_config.config_files.load_config_files", fake_load_config_files
    )
    monkeypatch.setattr("seamless_config.select.select_execution", fake_select_execution)
    monkeypatch.setattr(
        "seamless_config.select.get_selected_cluster", fake_get_selected_cluster
    )
    monkeypatch.setattr("seamless_config.select.get_execution", fake_get_execution)
    monkeypatch.setattr("seamless_config.set_workdir", lambda workdir: None)
    monkeypatch.setattr("seamless_config.change_stage", fake_change_stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seamless-run",
            "--dry",
            "--write-remote-job",
            "/users/sdevries/TESTDIR",
            "echo",
            "hello",
        ],
    )

    with pytest.raises(RuntimeError, match="stop-after-config"):
        cmd_main._main()

    assert select_execution_calls == []
