"""Command-line seamless-download executable script."""

from __future__ import annotations

import argparse
import json
import os
import sys

import seamless
import seamless.config as seamless_config
from seamless import Buffer, Checksum
from seamless.caching.buffer_cache import get_buffer_cache

from seamless_transformer.cmd.bytes2human import human2bytes
from seamless_transformer.cmd.download import download
from seamless_transformer.cmd.file_load import read_checksum_file, strip_textdata
from seamless_transformer.cmd.message import (
    set_header,
    set_verbosity,
    message as msg,
    message_and_exit as err,
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


def _get_buffer_content(checksum: Checksum) -> bytes | None:
    cache = get_buffer_cache()
    buffer = cache.get(checksum)
    if buffer is None:
        buffer = checksum.resolve()
    if buffer is None:
        return None
    return buffer.content


def _main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        prog="seamless-download",
        description="Download buffers from a remote buffer folder/server",
    )

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

    parser.add_argument(
        "-y",
        "--yes",
        dest="auto_confirm",
        help="""Sets any confirmation values to 'yes' automatically. Users will not be asked to confirm any file download.
    Downloads will happen without confirmation for up to 2000 files and up to 500 MB in total.
    These thresholds can be controlled by the environment variables:
    SEAMLESS_MAX_DOWNLOAD_FILES, SEAMLESS_MAX_DOWNLOAD_SIZE.""",
        action="store_const",
        const="yes",
    )

    parser.add_argument(
        "-n",
        "--no",
        dest="auto_confirm",
        help="""Sets any confirmation values to 'no' automatically. Users will not be asked to confirm any file download.
    Downloads will happen without confirmation for up to 2000 files and up to 500 MB in total.
    These thresholds can be controlled by the environment variables:
    SEAMLESS_MAX_DOWNLOAD_FILES, SEAMLESS_MAX_DOWNLOAD_SIZE.""",
        action="store_const",
        const="no",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="outputs",
        help="Explicitly specify output file or directory. Can be repeated in case of multiple downloads",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--stdout",
        help="Print all downloaded buffers to standard output",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--directory",
        help="Treat all raw checksum arguments as checksums to directory index buffers",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--index",
        dest="index_only",
        help="For directories (deep buffers), only download the index, and write one checksum file per buffer.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-v",
        dest="verbosity",
        help="""Verbose mode.
    Multiple -v options increase the verbosity. The maximum is 3""",
        action="count",
        default=0,
    )

    parser.add_argument(
        "files_directories_and_checksums",
        nargs=argparse.REMAINDER,
        help="files/directories/checksums that define the buffers to download",
    )

    args = parser.parse_args(argv)

    if not len(args.files_directories_and_checksums):
        print("At least one file, directory or checksum is required", file=sys.stderr)
        parser.print_usage(file=sys.stderr)
        return 1

    for path in args.files_directories_and_checksums:
        if path.startswith("-"):
            err("Options must be specified before files/directories/checksums")

    set_header("seamless-download")
    verbosity = min(args.verbosity, 3)
    set_verbosity(verbosity)

    if args.project:
        if select_project is None:
            err("seamless_config is unavailable; cannot set --project")
        try:
            project, subproject = _parse_scoped_value(args.project, "project")
            select_project(project)
            if subproject:
                select_subproject(subproject)
        except Exception as exc:
            err(str(exc))
    if args.stage:
        try:
            stage, substage = _parse_scoped_value(args.stage, "stage")
            if substage:
                seamless_config.set_stage(stage, substage, workdir=os.getcwd())
            else:
                seamless_config.set_stage(stage, workdir=os.getcwd())
        except Exception as exc:
            err(str(exc))

    max_download_files = os.environ.get("SEAMLESS_MAX_DOWNLOAD_FILES", "2000")
    max_download_files = int(max_download_files)
    max_download_size = os.environ.get("SEAMLESS_MAX_DOWNLOAD_SIZE", "500 MB")
    max_download_size = human2bytes(max_download_size)

    from seamless_config.extern_clients import set_remote_clients_from_env

    if not set_remote_clients_from_env(include_dask=False):

        seamless_config.init(workdir=os.getcwd())
        from seamless_config.select import get_selected_cluster

        if get_selected_cluster() is None:
            print(f"Cannot download without a cluster defined", file=sys.stderr)
            exit(1)

    ################################################################

    to_download = {}
    directories = []
    files = []
    index_checksums = {}
    paths = [path.rstrip(os.sep) for path in args.files_directories_and_checksums]
    for pathnr, path in enumerate(paths):
        parsed_checksum = None
        if not path.endswith(".INDEX"):
            if path.endswith(".CHECKSUM"):
                path2 = os.path.splitext(path)[0] + ".INDEX"
            else:
                try:
                    parsed_checksum = Checksum(path)
                except ValueError:
                    pass
                path2 = path + ".INDEX"
            if os.path.exists(path2) or (
                args.index_only
                and os.path.exists(os.path.splitext(path2)[0] + ".CHECKSUM")
            ):
                path = path2
            elif args.directory and parsed_checksum:
                path += ".INDEX"

        if path.endswith(".INDEX"):
            dirname = os.path.splitext(path)[0]
            if pathnr < len(args.outputs):
                dirname = args.outputs[pathnr]

            directories.append(dirname)
            index_checksum = None
            index_buffer_content = None
            if parsed_checksum:
                index_checksum = parsed_checksum
            else:
                if not os.path.exists(path):
                    index_err = f"Cannot read index file '{path}'"
                else:
                    with open(path) as f:
                        data = f.read()
                    data = strip_textdata(data)
                    index_buffer_content = data.encode() + b"\n"
                    if not index_buffer_content.strip(b"\n"):
                        index_buffer_content = None
                        index_err = f"Index file '{path}' is empty"
            if index_buffer_content is None:
                checksum_file = os.path.splitext(path)[0] + ".CHECKSUM"
                if not (parsed_checksum or args.directory) and not os.path.exists(
                    checksum_file
                ):
                    err(
                        f"{index_err}, {checksum_file} does not exist"  # pylint: disable=used-before-assignment
                    )
                if index_checksum is None:
                    index_checksum = read_checksum_file(checksum_file)
                if index_checksum is None:
                    err(f"{index_err}, {checksum_file} does not contain a checksum")
                assert index_checksum is not None
                index_checksum = Checksum(index_checksum)
                if not (args.index_only or args.directory):
                    msg(0, f"{index_err}, downloading from checksum ...")
                index_buffer_content = _get_buffer_content(index_checksum)
                if index_buffer_content is None:
                    if parsed_checksum:
                        err(f"Cannot download index buffer for {parsed_checksum}")
                    err(
                        f"{index_err}, cannot download checksum in {checksum_file}, CacheMissError"
                    )
                else:
                    if not (args.index_only or args.directory):
                        msg(0, "... success")
                    if parsed_checksum:
                        maybe_err_msg = f"Buffer with checksum {parsed_checksum} is not a valid index buffer"
                    else:
                        with open(path, "wb") as f:
                            f.write(index_buffer_content)
                        maybe_err_msg = f"{index_err}, but {checksum_file} does not contain the checksum of a valid directory index"

            else:
                index_checksum = Buffer(index_buffer_content).get_checksum()
                maybe_err_msg = f"File '{path}' is not a valid index file"

            if dirname != os.path.splitext(path)[0]:
                if index_buffer_content is not None:
                    with open(dirname + ".INDEX", "wb") as f:
                        f.write(index_buffer_content)
                if args.index_only:
                    with open(dirname + ".CHECKSUM", "w") as f:
                        f.write(index_checksum.hex() + "\n")

            has_err = False
            try:
                index_data = json.loads(index_buffer_content.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                has_err = True
            if not has_err:
                if not isinstance(index_data, dict):
                    has_err = True
                else:
                    for k, cs in index_data.items():
                        try:
                            cs2 = Checksum(cs)
                            assert cs2.hex() is not None
                        except Exception:
                            has_err = True
                            break
            if has_err:
                err(maybe_err_msg)  # pylint: disable=possibly-used-before-assignment
            else:
                for k, cs in index_data.items():
                    kk = os.path.join(dirname, k)
                    to_download[kk] = cs
            index_checksums[dirname] = index_checksum.hex()
            continue

        checksum = None
        if path.endswith(".CHECKSUM"):
            path = os.path.splitext(path)[0]
        elif parsed_checksum:
            checksum = parsed_checksum

        if checksum is None:
            checksum_file = path + ".CHECKSUM"
            checksum = read_checksum_file(checksum_file)
            if checksum is None:
                err(f"File '{checksum_file}' does not contain a checksum")
            checksum = Checksum(checksum)

        if pathnr < len(args.outputs):
            path = args.outputs[pathnr]

        to_download[path] = checksum.hex()
        files.append(path)

    ################################################################

    removed_files = []
    if not args.index_only:
        for directory in directories:
            if os.path.exists(directory):
                existing_files = [
                    os.path.join(dirpath, f)
                    for (dirpath, _, filenames) in os.walk(directory)
                    for f in filenames
                ]
                for f in existing_files:
                    if f not in to_download:
                        os.remove(f)
                        removed_files.append(f)
    if len(removed_files):
        msg(2, f"Removed {len(removed_files)} extra files in download directories")

    newdirs = {os.path.dirname(k) for k in to_download}
    for directory in directories:
        newdirs.add(directory)
    for newdir in newdirs:
        if len(newdir):
            os.makedirs(os.path.join(newdir), exist_ok=True)

    ################################################################

    if args.index_only:
        for path, checksum in to_download.items():
            with open(path + ".CHECKSUM", "w") as f:
                f.write(checksum + "\n")
    elif args.stdout:
        if len(directories):
            err("Cannot download and print directory to stdout")
        if len(files) > 1:
            err("Cannot download and print multiple files to stdout")
        else:
            cs = to_download[files[0]]
            file_buffer = _get_buffer_content(Checksum(cs))
            if file_buffer is None:
                err(f"Cannot download contents of file '{files[0]}', CacheMissError")
            sys.stdout.buffer.write(file_buffer)
    else:
        download(
            files,
            directories,
            checksum_dict=to_download,
            index_checksums=index_checksums,
            max_download_size=max_download_size,
            max_download_files=max_download_files,
            auto_confirm=args.auto_confirm,
        )


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    finally:
        seamless.close()


if __name__ == "__main__":
    sys.exit(main())
