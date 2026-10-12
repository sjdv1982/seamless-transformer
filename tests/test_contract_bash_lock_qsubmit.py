"""Queued submission notifies a bash parent even though no computation starts."""

import json
import os

from seamless_transformer.cmd.api import main as cmd_main
from seamless_transformer.process_lock import NOTIFY_ENV


def test_qsubmit_notifies_parent_fifo_before_returning_without_compute(monkeypatch, tmp_path):
    import seamless_remote.database_remote as database_remote
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cmd_main, "_require_config_file", lambda workdir: None)
    monkeypatch.setattr("seamless_config.set_workdir", lambda workdir: None)
    monkeypatch.setattr("seamless_config.change_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr("seamless_config.config_files.load_config_files", lambda: None)
    monkeypatch.setattr("seamless_config.select.select_execution", lambda mode: None)
    monkeypatch.setattr("seamless_config.select.get_selected_cluster", lambda: "test-cluster")
    monkeypatch.setattr("seamless_config.extern_clients.set_remote_clients_from_env", lambda **kwargs: False)
    monkeypatch.setattr(database_remote, "DISABLED", database_remote.DISABLED)
    checksum = "a" * 64
    monkeypatch.setattr(cmd_main, "prepare_bash_transformation", lambda *args, **kwargs: (checksum, {"__language__": "bash"}))
    queue = tmp_path / "queue"
    fifo = tmp_path / "notify"
    os.mkfifo(fifo)
    descriptor = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
    monkeypatch.setenv(NOTIFY_ENV, str(fifo))
    monkeypatch.setenv("SEAMLESS_QUEUE_FILE", str(queue))
    try:
        assert cmd_main._main(["--qsubmit", "-cp", ":output.txt", "echo", "queued"]) == 0
        assert len(os.read(descriptor, 1024)) == 1
        record = json.loads(queue.read_bytes().rstrip(b"\x00"))
        assert record["queue_command"] == "SUBMIT"
        assert record["transformation_checksum"] == checksum
        assert record["result_targets"]
        assert not (tmp_path / "output.txt").exists()
    finally:
        os.close(descriptor)
