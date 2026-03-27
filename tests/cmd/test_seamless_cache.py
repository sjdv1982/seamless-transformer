import sys

import seamless_config
import seamless_config.select as select

from seamless_transformer.cmd.api import main as cmd_main


def _reset_config_state(monkeypatch):
    monkeypatch.setattr(seamless_config, "_initialized", False)
    monkeypatch.setattr(seamless_config, "_set_workdir_called", False)
    monkeypatch.setattr(seamless_config, "_remote_clients_set", False)
    monkeypatch.setattr(seamless_config, "_workdir", None)

    monkeypatch.setattr(select, "_current_cluster", None)
    monkeypatch.setattr(select, "_current_project", None)
    monkeypatch.setattr(select, "_current_subproject", None)
    monkeypatch.setattr(select, "_current_stage", None)
    monkeypatch.setattr(select, "_current_substage", None)
    monkeypatch.setattr(select, "_current_execution", "process")
    monkeypatch.setattr(select, "_execution_source", None)
    monkeypatch.setattr(select, "_execution_command_seen", False)
    monkeypatch.setattr(select, "_current_persistent", None)
    monkeypatch.setattr(select, "_persistent_source", None)
    monkeypatch.setattr(select, "_persistent_command_seen", False)
    monkeypatch.setattr(select, "_current_queue", None)
    monkeypatch.setattr(select, "_queue_source", None)
    monkeypatch.setattr(select, "_queue_cluster", None)
    monkeypatch.setattr(select, "_current_remote", None)
    monkeypatch.setattr(select, "_remote_source", None)


def test_seamless_run_dry_uses_seamless_cache_without_yaml(monkeypatch, tmp_path):
    _reset_config_state(monkeypatch)
    workdir = tmp_path / "demo-project"
    workdir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    monkeypatch.chdir(workdir)
    monkeypatch.setenv("SEAMLESS_CACHE", str(cache_dir))
    monkeypatch.setattr(sys, "argv", ["seamless-run", "--dry", "echo", "hello"])

    assert cmd_main._main() == 0
    assert select.get_selected_cluster() == "__SEAMLESS_CACHE__"
    cluster, project, subproject, stage, substage = select.get_current()
    assert cluster == "__SEAMLESS_CACHE__"
    assert project == select.PROJECT_TOPLEVEL
    assert subproject is None
    assert stage is None
    assert substage is None
