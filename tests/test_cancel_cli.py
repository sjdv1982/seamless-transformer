from seamless import Checksum

from seamless_transformer.api import cancel


def test_cancel_cli_returns_zero_when_backend_cancels(monkeypatch, capsys):
    checksum = Checksum("3" * 64)
    calls = []

    monkeypatch.setattr(cancel, "_parse_checksum", lambda _arg: checksum)
    monkeypatch.setattr(cancel.seamless_config, "init", lambda **_kwargs: None)

    def fake_cancel_by_checksum(tf_checksum):
        calls.append(tf_checksum)
        return True, ["dask: canceled"]

    monkeypatch.setattr(cancel, "cancel_by_checksum", fake_cancel_by_checksum)

    assert cancel._main([checksum.hex()]) == 0
    assert calls == [checksum]
    assert capsys.readouterr().out.strip() == "dask: canceled"


def test_cancel_cli_returns_two_when_nothing_active(monkeypatch, capsys):
    checksum = Checksum("4" * 64)

    monkeypatch.setattr(cancel, "_parse_checksum", lambda _arg: checksum)
    monkeypatch.setattr(cancel.seamless_config, "init", lambda **_kwargs: None)
    monkeypatch.setattr(
        cancel,
        "cancel_by_checksum",
        lambda _checksum: (False, ["dask: not-running"]),
    )

    assert cancel._main([checksum.hex()]) == 2
    assert capsys.readouterr().out.strip() == "dask: not-running"


def test_cancel_by_checksum_uses_process_registry(monkeypatch):
    checksum = Checksum("5" * 64)

    class FakeCache:
        def cancel_by_checksum(self, tf_checksum):
            assert tf_checksum == checksum
            return True

    monkeypatch.setattr(
        "seamless_transformer.transformation_cache.get_transformation_cache",
        lambda: FakeCache(),
    )
    monkeypatch.setattr(
        "seamless_dask.transformer_client.get_seamless_dask_client",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(cancel, "jobserver_remote", None, raising=False)

    canceled, messages = cancel.cancel_by_checksum(checksum)

    assert canceled is True
    assert "process: canceled" in messages
