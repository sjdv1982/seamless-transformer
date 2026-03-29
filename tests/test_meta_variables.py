import hashlib

import pytest

from seamless_transformer.cmd.api import main as cmd_main
from seamless_transformer.cmd.bash_transformation import (
    prepare_bash_code,
    prepare_bash_transformation,
)


def _fake_register_buffer(buffer, dry_run=False):
    return hashlib.sha256(bytes(buffer)).hexdigest()


def test_prepare_bash_code_exports_meta_variables():
    bashcode = prepare_bash_code(
        "echo $THREADS",
        make_executables=[],
        result_targets=None,
        capture_stdout=True,
        meta_variable_names=["THREADS", "DEBUG"],
    )

    assert (
        bashcode
        == 'export DEBUG="${META__DEBUG}"\n'
        'export THREADS="${META__THREADS}"\n'
        "(\n"
        "echo $THREADS\n"
        ") > RESULT"
    )


def test_meta_variable_values_do_not_change_transformation_checksum(monkeypatch):
    monkeypatch.setattr(
        "seamless_transformer.cmd.bash_transformation.register_buffer",
        _fake_register_buffer,
    )
    monkeypatch.setattr(
        "seamless_transformer.cmd.bash_transformation.register_dict",
        lambda value, dry_run=False: hashlib.sha256(repr(value).encode()).hexdigest(),
    )

    common_kwargs = dict(
        code="echo $THREADS",
        checksum_dict={},
        directories=[],
        make_executables=[],
        result_targets=None,
        capture_stdout=True,
        environment={},
        meta={},
        variables=None,
        dry_run=True,
    )

    checksum1, transformation1 = prepare_bash_transformation(
        meta_variables={"THREADS": ("2", "str")},
        **common_kwargs,
    )
    checksum2, transformation2 = prepare_bash_transformation(
        meta_variables={"THREADS": ("8", "str")},
        **common_kwargs,
    )

    assert checksum1 == checksum2
    assert transformation1["code"][2] == transformation2["code"][2]
    assert transformation1["META__THREADS"][2] != transformation2["META__THREADS"][2]


def test_meta_variable_names_change_generated_code(monkeypatch):
    monkeypatch.setattr(
        "seamless_transformer.cmd.bash_transformation.register_buffer",
        _fake_register_buffer,
    )

    common_kwargs = dict(
        code="echo ok",
        checksum_dict={},
        directories=[],
        make_executables=[],
        result_targets=None,
        capture_stdout=True,
        environment={},
        meta={},
        variables=None,
        dry_run=True,
    )

    checksum1, transformation1 = prepare_bash_transformation(
        meta_variables={"DEBUG": ("1", "str")},
        **common_kwargs,
    )
    checksum2, transformation2 = prepare_bash_transformation(
        meta_variables={"THREADS": ("1", "str")},
        **common_kwargs,
    )

    assert checksum1 != checksum2
    assert transformation1["code"][2] != transformation2["code"][2]


def test_parse_metavar_spec_rejects_internal_prefix(monkeypatch):
    monkeypatch.setattr(
        cmd_main,
        "err",
        lambda message: (_ for _ in ()).throw(RuntimeError(message)),
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid --metavar 'META__DEBUG=1': name must not start with META__",
    ):
        cmd_main._parse_metavar_spec("META__DEBUG=1")
