"""Command-line seamless-run-transformation executable script."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

import seamless
import seamless.config as seamless_config
from seamless import CacheMissError, Checksum

from seamless_transformer.cmd.file_load import read_checksum_file
from seamless_transformer.transformation_class import (
    compute_transformation_sync,
    transformation_from_dict,
)

try:
    from seamless_config.select import select_project, select_subproject
except Exception:  # pragma: no cover - optional dependency
    select_project = None
    select_subproject = None


def _parse_scoped_value(value: str, label: str) -> tuple[str, str | None]:
    if not value:
        raise ValueError(f"--{label} requires a value")
    if ":" in value:
        head, tail = value.split(":", 1)
        if not head:
            raise ValueError(f"Invalid --{label} value '{value}'")
        if tail == "":
            tail = None
        return head, tail
    return value, None


def _parse_checksum(checksum_arg: str) -> Checksum:
    if checksum_arg.endswith(".CHECKSUM") and os.path.exists(checksum_arg):
        checksum_hex = read_checksum_file(checksum_arg)
        if not checksum_hex:
            raise ValueError(f"Invalid checksum file '{checksum_arg}'")
        return Checksum(checksum_hex)
    return Checksum(checksum_arg)


def _resolve_transformation_dict(checksum: Checksum) -> dict[str, Any]:
    transformation = checksum.resolve(celltype="plain")
    if transformation is None:
        raise CacheMissError(checksum)
    if not isinstance(transformation, dict):
        raise TypeError(f"Transformation buffer is not a dict: {type(transformation)}")
    if "__output__" not in transformation:
        transformation["__output__"] = ("result", "mixed", None)
    return transformation


def _extract_dunder(transformation_dict: dict[str, Any]) -> dict[str, Any]:
    core_keys = {"__language__", "__output__", "__as__", "__format__"}
    return {
        k: v
        for k, v in transformation_dict.items()
        if k.startswith("__") and k not in core_keys and not k.startswith("__code")
    }


def _configure_logging(*, debug: bool, verbose: bool) -> None:
    logging.basicConfig()
    if debug:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.set_debug(True)
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.ERROR
    logging.getLogger("seamless").setLevel(level)
    logging.getLogger("seamless_transformer").setLevel(level)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="seamless-run-transformation",
        description="Run a transformation from checksum",
    )
    parser.add_argument("checksum", help="Seamless checksum or checksum file")
    parser.add_argument(
        "--project",
        metavar="PROJECT[:SUBPROJECT]",
        help="set Seamless project (and subproject). Each project has independent storage",
    )
    parser.add_argument(
        "--stage",
        metavar="STAGE[:SUBSTAGE]",
        help="set Seamless project stage (and substage). Each project stage has independent storage",
    )

    # TODO: restore when ncores support lands in the new code base.
    # parser.add_argument("--ncores", type=int, default=None)

    parser.add_argument("--direct-print", dest="direct_print", action="store_true")
    parser.add_argument(
        "--verbose",
        help="Verbose mode, setting the Seamless logger to INFO",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        help="Debugging mode. Turns on asyncio debugging, and sets the Seamless logger to DEBUG",
        action="store_true",
    )

    parser.add_argument(
        "--scratch",
        help="Don't write the computed result buffer.",
        default=False,
        action="store_true",
    )
    parser.add_argument("--output", help="Output file (default: stdout)")

    # TODO: --dunder, --fingertip, --undo, --global-info need new-codebase support.
    # parser.add_argument("--dunder", help="Dunder file with transformation metadata")
    # parser.add_argument("--fingertip", action="store_true")
    # parser.add_argument("--undo", action="store_true", default=False)
    # parser.add_argument("--global-info", dest="global_info")

    args = parser.parse_args(argv)

    _configure_logging(debug=args.debug, verbose=args.verbose)
    if args.project:
        if select_project is None:
            print(
                "seamless_config is unavailable; cannot set --project", file=sys.stderr
            )
            return 1
        try:
            project, subproject = _parse_scoped_value(args.project, "project")
            select_project(project)
            if subproject:
                select_subproject(subproject)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if args.stage:
        try:
            stage, substage = _parse_scoped_value(args.stage, "stage")
            if substage:
                seamless_config.set_stage(stage, substage, workdir=os.getcwd())
            else:
                seamless_config.set_stage(stage, workdir=os.getcwd())
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
    seamless_config.init(workdir=os.getcwd())

    try:
        checksum = _parse_checksum(args.checksum)
        transformation_dict = _resolve_transformation_dict(checksum)
    except Exception as exc:
        import traceback

        excs = traceback.format_exc(limit=0)
        print(excs, file=sys.stderr)
        return 1

    if args.direct_print:
        meta = transformation_dict.get("__meta__")
        if not isinstance(meta, dict):
            meta = {}
        meta["__direct_print__"] = True
        transformation_dict["__meta__"] = meta

    tf_dunder = _extract_dunder(transformation_dict)
    try:
        transformation = transformation_from_dict(
            transformation_dict,
            meta={},
            scratch=bool(args.scratch),
            tf_dunder=tf_dunder,
        )
        result_checksum = compute_transformation_sync(
            transformation,
            require_value=False,
        )
        if result_checksum is None:
            raise RuntimeError("Result checksum unavailable")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    result_hex = Checksum(result_checksum).hex()
    if args.output:
        try:
            with open(args.output, "w") as output_file:
                print(result_hex, file=output_file)
        except Exception as exc:
            print(f"Cannot write output file '{args.output}': {exc}", file=sys.stderr)
            return 1
    else:
        print(result_hex)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    finally:
        seamless.close()


if __name__ == "__main__":
    sys.exit(main())
