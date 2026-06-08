from seamless import Checksum

from seamless_transformer.api import run_transformation


def test_run_transformation_cli_wires_strict(monkeypatch, capsys):
    checksum = Checksum("1" * 64)
    calls = []

    monkeypatch.setattr(run_transformation, "_parse_checksum", lambda _arg: checksum)
    monkeypatch.setattr(
        run_transformation,
        "_resolve_transformation_dict",
        lambda _checksum: {"__output__": ("result", "mixed", None)},
    )

    def fake_run_sync(
        transformation_dict,
        *,
        tf_checksum,
        tf_dunder,
        scratch,
        require_value,
        strict_dunder=False,
    ):
        calls.append(
            {
                "transformation_dict": transformation_dict,
                "tf_checksum": tf_checksum,
                "tf_dunder": tf_dunder,
                "scratch": scratch,
                "require_value": require_value,
                "strict_dunder": strict_dunder,
            }
        )
        return Checksum("2" * 64)

    monkeypatch.setattr(run_transformation, "run_sync", fake_run_sync)

    assert run_transformation._main(["--strict", checksum.hex()]) == 0
    assert calls[0]["strict_dunder"] is True
    assert calls[0]["scratch"] is False
    assert calls[0]["require_value"] is False
    assert capsys.readouterr().out.strip() == "2" * 64
