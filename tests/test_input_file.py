from pathlib import Path

from seamless_transformer.cmd.api import main as main_mod
from seamless_transformer.cmd.file_mapping import get_file_mapping
from seamless_transformer.cmd.parsing import guess_arguments


def test_input_file_entries_remain_relative_to_listfile_when_combined_with_manual_inputs(
    tmp_path, monkeypatch
):
    workdir = tmp_path / "workdir"
    depsdir = workdir / "deps"
    manualdir = workdir / "manual"
    workdir.mkdir()
    depsdir.mkdir()
    manualdir.mkdir()

    listfile = depsdir / "inputs.txt"
    first_dep = depsdir / "alpha.py"
    second_dep = depsdir / "nested" / "beta.h"
    second_dep.parent.mkdir()
    manual_input = manualdir / "gamma.py"

    listfile.write_text("alpha.py\nnested/beta.h\n", encoding="utf-8")
    first_dep.write_text("print('alpha')\n", encoding="utf-8")
    second_dep.write_text("#define BETA 1\n", encoding="utf-8")
    manual_input.write_text("print('gamma')\n", encoding="utf-8")

    monkeypatch.chdir(workdir)

    extra_inputs = [manual_input.relative_to(workdir).as_posix()]
    extra_inputs.extend(main_mod._load_input_file(listfile.relative_to(workdir).as_posix()))

    argtypes = guess_arguments(
        extra_inputs,
        overrule_ext=True,
        overrule_no_ext=True,
    )
    mapped = get_file_mapping(
        argtypes,
        mapping_mode="literal",
        working_directory=workdir.as_posix(),
    )

    assert set(mapped) == {
        "@order",
        "manual/gamma.py",
        "deps/alpha.py",
        "deps/nested/beta.h",
    }
    assert mapped["manual/gamma.py"]["mapping"] == str(manual_input.resolve())
    assert mapped["deps/alpha.py"]["mapping"] == str(first_dep.resolve())
    assert mapped["deps/nested/beta.h"]["mapping"] == str(second_dep.resolve())


def test_multiple_input_files_all_contribute_and_stay_relative_to_their_own_listfile(
    tmp_path, monkeypatch
):
    workdir = tmp_path / "workdir"
    first_dir = workdir / "first"
    second_dir = workdir / "second"
    workdir.mkdir()
    first_dir.mkdir()
    second_dir.mkdir()

    first_listfile = first_dir / "inputs-a.txt"
    second_listfile = second_dir / "inputs-b.txt"
    first_dep = first_dir / "alpha.py"
    second_dep = second_dir / "beta.h"

    first_dep.write_text("print('alpha')\n", encoding="utf-8")
    second_dep.write_text("#define BETA 1\n", encoding="utf-8")
    first_listfile.write_text("alpha.py\n", encoding="utf-8")
    second_listfile.write_text("beta.h\n", encoding="utf-8")

    monkeypatch.chdir(workdir)

    extra_inputs = []
    for input_file in (
        first_listfile.relative_to(workdir).as_posix(),
        second_listfile.relative_to(workdir).as_posix(),
    ):
        extra_inputs.extend(main_mod._load_input_file(input_file))

    argtypes = guess_arguments(
        extra_inputs,
        overrule_ext=True,
        overrule_no_ext=True,
    )
    mapped = get_file_mapping(
        argtypes,
        mapping_mode="literal",
        working_directory=workdir.as_posix(),
    )

    assert set(mapped) == {"@order", "first/alpha.py", "second/beta.h"}
    assert mapped["first/alpha.py"]["mapping"] == str(first_dep.resolve())
    assert mapped["second/beta.h"]["mapping"] == str(second_dep.resolve())


def test_input_file_entries_follow_the_real_listfile_directory_when_accessed_via_symlink(
    tmp_path, monkeypatch
):
    real_root = tmp_path / "real"
    workdir = tmp_path / "workdir"
    code_dir = real_root / "code"
    fraglib_dir = real_root / "fraglib"
    workdir.mkdir()
    code_dir.mkdir(parents=True)
    fraglib_dir.mkdir(parents=True)

    listfile = code_dir / "inputs.txt"
    dep = fraglib_dir / "value.txt"
    dep.write_text("payload\n", encoding="utf-8")
    listfile.write_text("../fraglib/value.txt\n", encoding="utf-8")
    (workdir / "code").symlink_to(code_dir, target_is_directory=True)

    monkeypatch.chdir(workdir)

    extra_inputs = main_mod._load_input_file("code/inputs.txt")
    assert extra_inputs == ["code/../fraglib/value.txt"]

    argtypes = guess_arguments(
        extra_inputs,
        overrule_ext=True,
        overrule_no_ext=True,
    )
    mapped = get_file_mapping(
        argtypes,
        mapping_mode="literal",
        working_directory=workdir.as_posix(),
    )

    assert mapped["fraglib/value.txt"]["mapping"].endswith(
        "/workdir/code/../fraglib/value.txt"
    )
