import seamless_config
import seamless_config.select as select

from seamless_transformer.cmd.api import main as cmd_main
import seamless_transformer.probe_capture as probe_capture


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
    monkeypatch.setattr(select, "_current_record", False)
    monkeypatch.setattr(select, "_record_source", None)
    monkeypatch.setattr(select, "_record_command_seen", False)
    monkeypatch.setattr(select, "_current_node", None)
    monkeypatch.setattr(select, "_node_source", None)


def test_probe_main_accepts_naked_probe(monkeypatch, tmp_path):
    _reset_config_state(monkeypatch)
    workdir = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    workdir.mkdir()
    cache_dir.mkdir()
    calls = []

    def fake_refresh(
        transformation_dict,
        target_tf_dunder=None,
        *,
        force=False,
        msg_func=None,
    ):
        calls.append(
            {
                "transformation_dict": transformation_dict,
                "target_tf_dunder": target_tf_dunder,
                "force": force,
                "msg_func": msg_func,
            }
        )
        return {"refreshed": [], "reused": []}

    monkeypatch.chdir(workdir)
    monkeypatch.setenv("SEAMLESS_CACHE", str(cache_dir))
    monkeypatch.setattr(probe_capture, "refresh_required_buckets_sync", fake_refresh)

    assert cmd_main.probe_main([]) == 0
    assert len(calls) == 1
    assert calls[0]["target_tf_dunder"] is not None
    assert calls[0]["force"] is False


def test_probe_main_wires_target_tf_dunder_and_force(monkeypatch, tmp_path):
    _reset_config_state(monkeypatch)
    workdir = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    workdir.mkdir()
    cache_dir.mkdir()
    calls = []

    def fake_refresh(
        transformation_dict,
        target_tf_dunder=None,
        *,
        force=False,
        msg_func=None,
    ):
        calls.append((transformation_dict, target_tf_dunder, force, msg_func))
        return {
            "refreshed": [{"bucket_kind": "node"}],
            "reused": [{"bucket_kind": "environment"}],
        }

    monkeypatch.chdir(workdir)
    monkeypatch.setenv("SEAMLESS_CACHE", str(cache_dir))
    monkeypatch.setattr(probe_capture, "refresh_required_buckets_sync", fake_refresh)

    assert cmd_main.probe_main(["--force", "true"]) == 0
    assert len(calls) == 1
    assert calls[0][1] is not None
    assert calls[0][2] is True
