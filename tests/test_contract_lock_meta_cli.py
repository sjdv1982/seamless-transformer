"""seamless-run lock flags reach bash transformation metadata."""

import pytest

from seamless_transformer.cmd.api import main as cmd_main
from seamless_transformer.transformation_utils import resolve_lock_meta


@pytest.mark.parametrize("flags, expected", [
    ([], (True, False)),
    (["--driver"], (False, True)),
    (["--no-process-lock"], (False, False)),
    (["--driver", "--no-process-lock"], (False, True)),
])
def test_seamless_run_lock_flags_reach_bash_builder(monkeypatch, tmp_path, flags, expected):
    import seamless_remote.database_remote as database_remote

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cmd_main, "_require_config_file", lambda workdir: None)
    monkeypatch.setattr("seamless_config.set_workdir", lambda workdir: None)
    monkeypatch.setattr("seamless_config.change_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr("seamless_config.config_files.load_config_files", lambda: None)
    monkeypatch.setattr("seamless_config.select.select_execution", lambda mode: None)
    monkeypatch.setattr("seamless_config.select.get_selected_cluster", lambda: "test-cluster")
    monkeypatch.setattr(
        "seamless_config.extern_clients.set_remote_clients_from_env",
        lambda **kwargs: False,
    )
    # Restore this setting when the CLI dry-run branch changes it.
    monkeypatch.setattr(database_remote, "DISABLED", database_remote.DISABLED)
    captured = {}

    class Prepared(Exception):
        pass

    def capture_prepare(*args, **kwargs):
        captured.update(kwargs)
        raise Prepared()

    monkeypatch.setattr(cmd_main, "prepare_bash_transformation", capture_prepare)
    with pytest.raises(Prepared):
        cmd_main._main(["--dry", "--write-remote-job", "/remote/job", *flags, "echo", "hello"])
    meta = captured["meta"]
    assert resolve_lock_meta(meta) == expected
    assert meta[cmd_main.REMOTE_JOB_META_KEY] == "/remote/job"
    if "--driver" in flags:
        assert meta["driver"] is True
        assert meta["process_lock"] is False
    elif "--no-process-lock" in flags:
        assert meta["process_lock"] is False
    assert captured["dry_run"] is False  # remote-job mode implies upload
