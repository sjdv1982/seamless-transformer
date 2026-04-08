import subprocess
from pathlib import Path

from seamless_transformer.cmd.parsing import fill_checksum_arguments, guess_arguments


def test_explicit_checksum_argument_is_treated_as_literal_file(tmp_path):
    checksum_file = tmp_path / "input.dat.CHECKSUM"
    checksum_file.write_text("literal sidecar file\n", encoding="utf-8")

    result = guess_arguments([checksum_file.as_posix()])

    assert result["@order"] == [checksum_file.as_posix()]
    assert result[checksum_file.as_posix()]["type"] == "file"
    assert "checksum" not in result[checksum_file.as_posix()]


def test_implicit_checksum_sidecar_is_still_used_for_base_name(tmp_path):
    target = tmp_path / "input.dat"
    checksum_file = Path(str(target) + ".CHECKSUM")
    checksum_file.write_text("0" * 64 + "\n", encoding="utf-8")

    result = guess_arguments([target.as_posix()])

    assert result[target.as_posix()]["type"] == "file"
    assert result[target.as_posix()]["checksum"] == "0" * 64


def test_fill_checksum_arguments_keeps_explicit_checksum_literal():
    files = ["input.dat.CHECKSUM"]
    order = files.copy()

    fill_checksum_arguments(files, order)

    assert files == ["input.dat.CHECKSUM"]
    assert order == ["input.dat.CHECKSUM"]


def test_seamlessify_strips_explicit_sidecars_before_preview():
    script = Path(__file__).resolve().parent.parent / "bin" / "_seamlessify"
    result = subprocess.run(
        [
            script.as_posix(),
            "python",
            "analyze.py",
            "data/input.dat.CHECKSUM",
            "results/outdir.INDEX",
            "@@@",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = result.stdout.strip()
    assert ".CHECKSUM" not in stdout
    assert ".INDEX" not in stdout
    assert "data/input.dat" in stdout
    assert "results/outdir" in stdout
