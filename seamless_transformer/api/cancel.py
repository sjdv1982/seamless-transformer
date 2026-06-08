"""Command-line seamless-cancel executable script."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import seamless
import seamless.config as seamless_config
from seamless import Checksum

from seamless_transformer.api.run_transformation import (
    _configure_logging,
    _parse_checksum,
    _parse_scoped_value,
)

try:
    from seamless_config.select import select_project, select_subproject
except Exception:  # pragma: no cover - optional dependency
    select_project = None
    select_subproject = None


def _configure_scope(args: argparse.Namespace) -> None:
    if args.project:
        if select_project is None:
            raise RuntimeError("seamless_config is unavailable; cannot set --project")
        project, subproject = _parse_scoped_value(args.project, "project")
        select_project(project)
        if subproject:
            select_subproject(subproject)
    if args.stage:
        stage, substage = _parse_scoped_value(args.stage, "stage")
        if substage:
            seamless_config.set_stage(stage, substage, workdir=os.getcwd())
        else:
            seamless_config.set_stage(stage, workdir=os.getcwd())


def cancel_by_checksum(tf_checksum: Checksum) -> tuple[bool, list[str]]:
    """Cancel active submissions for ``tf_checksum`` in the current namespace."""

    canceled = False
    messages: list[str] = []

    try:
        from seamless_transformer.transformation_cache import (
            get_transformation_cache,
        )
    except Exception as exc:
        messages.append(f"process: error: {exc}")
    else:
        try:
            if get_transformation_cache().cancel_by_checksum(tf_checksum):
                canceled = True
                messages.append("process: canceled")
            else:
                messages.append("process: not-running")
        except Exception as exc:
            messages.append(f"process: error: {exc}")

    try:
        from seamless_dask.transformer_client import get_seamless_dask_client
    except Exception:
        dask_client = None
    else:
        dask_client = get_seamless_dask_client()
    if dask_client is not None:
        cancel = getattr(dask_client, "cancel_by_checksum", None)
        if callable(cancel):
            try:
                if cancel(tf_checksum):
                    canceled = True
                    messages.append("dask: canceled")
                else:
                    messages.append("dask: not-running")
            except Exception as exc:
                messages.append(f"dask: error: {exc}")

    try:
        from seamless_remote import jobserver_remote
    except Exception:
        jobserver_remote = None
    if jobserver_remote is not None:
        cancel = getattr(jobserver_remote, "cancel_transformation", None)
        if callable(cancel):
            try:
                if cancel(tf_checksum):
                    canceled = True
                    messages.append("jobserver: canceled")
                else:
                    messages.append("jobserver: not-running")
            except RuntimeError as exc:
                if "No jobserver clients" not in str(exc):
                    messages.append(f"jobserver: error: {exc}")
            except Exception as exc:
                messages.append(f"jobserver: error: {exc}")

    if not messages:
        messages.append("no active cancellation backend")
    return canceled, messages


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="seamless-cancel",
        description="Cancel an active Seamless transformation by checksum",
    )
    parser.add_argument("checksum", help="Transformation checksum or checksum file")
    parser.add_argument(
        "--project",
        metavar="PROJECT[:SUBPROJECT]",
        help="set Seamless project (and subproject)",
    )
    parser.add_argument(
        "--stage",
        metavar="STAGE[:SUBSTAGE]",
        help="set Seamless project stage (and substage)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(debug=args.debug, verbose=args.verbose)
    logging.getLogger("seamless_transformer").debug("seamless-cancel starting")

    try:
        _configure_scope(args)
        seamless_config.init(workdir=os.getcwd())
        checksum = _parse_checksum(args.checksum)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    canceled, messages = cancel_by_checksum(checksum)
    for message in messages:
        print(message)
    return 0 if canceled else 2


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    finally:
        seamless.close()


if __name__ == "__main__":
    sys.exit(main())
