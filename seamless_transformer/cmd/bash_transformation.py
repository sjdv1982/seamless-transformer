"""Build a Seamless bash transformation from a parsed cmd-seamless command line"""

import os
import builtins

from seamless import Checksum
from seamless.checksum.serialize import serialize_sync as serialize

from seamless_transformer.transformation_utils import tf_get_buffer

from .message import message as msg
from .register import register_buffer, register_dict


def prepare_bash_code(
    code: str,
    *,
    make_executables: list[str],
    result_targets: dict | None,
    capture_stdout: bool,
):
    """Adapt cmd-seamless bash command into bash code for a bash transformation.
    Deals with:
    - If the command word refers to a file, make it executable
    - Add current directory to PATH
    - Handle stdout/stderr capture
    - Handle result capture
    """

    bashcode = ""
    if make_executables:
        has_cwd_executable = False
        make_executables_str = ""
        for make_executable in make_executables:
            if not os.path.dirname(make_executable):
                has_cwd_executable = True
            make_executables_str += f" '{make_executable}'"
        if has_cwd_executable:
            bashcode += "export PATH=./:$PATH\n"
        bashcode += f"chmod +x{make_executables_str}\n"

    if not result_targets:
        assert capture_stdout
        bashcode += "(\n" + code + "\n) > RESULT"
    else:
        code2 = code
        if capture_stdout:
            code2 = "(\n" + code + "\n) > STDOUT"
        mvcode = ""
        result_target_dirs = []
        for tar in result_targets:
            tardir = os.path.join("RESULT", os.path.dirname(tar))
            mvcode += f"mv {tar} {tardir}\n"
            if tardir not in result_target_dirs:
                result_target_dirs.append(tardir)
        bashcode += f"mkdir -p {' '.join(result_target_dirs)}\n"
        bashcode += code2 + "\n"
        bashcode += mvcode
    return bashcode


def prepare_bash_transformation(
    code: str,
    checksum_dict: dict[str, str],
    *,
    directories: list[str],
    make_executables: list[str],
    result_targets: dict | None,
    capture_stdout: bool,
    environment: dict,
    meta: dict,
    variables: dict,
    dry_run: bool = False,
) -> str:
    """Prepare a bash transformation for execution.

    Input:

    - code: bash code to execute inside a workspace. The code must write its result:
        - to /dev/stdout if result_mode is "stdout"
        - or: to a file called RESULT, if result_mode is "file"
        - or: to a directory called RESULT, if result_mode is "directory"
    - checksum_dict: checksums of the files/directories to be injected in the workspace
    - directories: list of the keys in checksum_dict that are directories
    - make_executables: list of paths where the executable bit must be set
    - capture_stdout
    - result_targets: server files containing results
    - environment
    - meta
    - variables: ....

    Returns: transformation checksum, transformation dict
    """
    bashcode = prepare_bash_code(
        code,
        make_executables=make_executables,
        result_targets=result_targets,
        capture_stdout=capture_stdout,
    )

    new_args = {
        "code": ("text", None, bashcode),
    }
    for attr in ("bashcode", "pins_"):
        if attr in checksum_dict:
            msg(0, f"'{attr}' cannot be in checksum dict")
            exit(1)

    transformation_dict = {"__language__": "bash"}

    if not result_targets:
        transformation_dict["__output__"] = ("result", "bytes", None)
    else:
        transformation_dict["__output__"] = ("result", "mixed", None, {"*": "##"})

    if meta:
        transformation_dict["__meta__"] = meta
    if environment:
        env_checksum = register_dict(environment, dry_run=dry_run)
        transformation_dict["__env__"] = env_checksum
    format_ = {}
    for k, v in checksum_dict.items():
        if not isinstance(v, str):
            v = Checksum(v).hex()
        if k in directories:
            fmt = {
                "filesystem": {"optional": True, "mode": "directory"},
                "hash_pattern": {"*": "##"},
            }
            transformation_dict[k] = "mixed", None, v
        else:
            fmt = {"filesystem": {"optional": True, "mode": "file"}}
            transformation_dict[k] = "bytes", None, v
        format_[k] = fmt

    if format_:
        transformation_dict["__format__"] = format_
    if variables:
        for k, (v, celltype) in variables.items():
            if celltype in ("int", "float", "bool", "str"):
                value = getattr(builtins, celltype)(v)
            else:
                raise TypeError(celltype)
            new_args[k] = celltype, None, value

    for k, v in new_args.items():
        celltype, subcelltype, value = v
        buffer = serialize(value, celltype)
        checksum_hex = register_buffer(buffer, dry_run=dry_run)
        vv = celltype, subcelltype, checksum_hex
        transformation_dict[k] = vv

    tf_buffer = tf_get_buffer(transformation_dict)
    tf_checksum = tf_buffer.get_checksum()
    tf_buffer.tempref()

    return tf_checksum, transformation_dict


def _extract_dunder(transformation_dict: dict) -> dict:
    core_keys = {"__language__", "__output__", "__as__", "__format__"}
    return {
        k: v
        for k, v in transformation_dict.items()
        if k.startswith("__")
        and k not in core_keys
        and not k.startswith("__code")
    }


def run_transformation(
    transformation_dict: dict, *, undo: bool, fingertip=False, scratch=False
):
    """Run a cmd-seamless transformation dict.
    First convert it into a bash transformation."""
    if undo:
        raise NotImplementedError("Undo is not supported in seamless-transformer yet")
    if scratch and fingertip:
        raise ValueError("Cannot require fingertip for scratch transformations")

    from seamless_transformer.transformation_class import (
        compute_transformation_sync,
        transformation_from_dict,
    )

    tf_dunder = _extract_dunder(transformation_dict)
    transformation = transformation_from_dict(
        transformation_dict,
        meta={},
        scratch=scratch,
        tf_dunder=tf_dunder,
    )
    result_checksum = compute_transformation_sync(
        transformation,
        require_value=bool(fingertip),
    )
    if result_checksum is None:
        raise RuntimeError("Result checksum unavailable")
    return Checksum(result_checksum)


async def run_transformation_async(
    transformation_dict: dict, *, undo: bool, fingertip=False, scratch=False
):
    """Run a cmd-seamless transformation dict.
    First convert it into a bash transformation."""
    if undo:
        raise NotImplementedError("Undo is not supported in seamless-transformer yet")
    if scratch and fingertip:
        raise ValueError("Cannot require fingertip for scratch transformations")

    from seamless_transformer.transformation_class import transformation_from_dict

    tf_dunder = _extract_dunder(transformation_dict)
    transformation = transformation_from_dict(
        transformation_dict,
        meta={},
        scratch=scratch,
        tf_dunder=tf_dunder,
    )
    result_checksum = await transformation.computation(
        require_value=bool(fingertip)
    )
    if result_checksum is None:
        raise RuntimeError("Result checksum unavailable")
    return Checksum(result_checksum)
