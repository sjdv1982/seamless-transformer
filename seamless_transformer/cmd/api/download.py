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
from seamless_transformer.cmd.register import _resolve_destination_path
from seamless_transformer.compression_utils import strip_compression_suffix

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


def _get_transfer_mode(args) -> str | None:
    if args.copy:
        return "copy"
    if args.hardlink:
        return "hardlink"
    if args.softlink:
        return "symlink"
    return None


def _get_existing_entry_policy(args) -> str:
    if args.dest_verify:
        return "verify"
    if args.dest_repair:
        return "repair"
    return "skip"


def _get_auto_destination_folder() -> str | None:
    try:
        import seamless_remote.buffer_remote as buffer_remote
    except Exception:
        return None
    try:
        extern_clients = buffer_remote.inspect_extern_clients()
    except Exception:
        extern_clients = []
    try:
        launched_clients = buffer_remote.inspect_launched_clients()
    except Exception:
        launched_clients = []
    buffer_clients = extern_clients + launched_clients
    candidates = [
        client.get("directory")
        for client in buffer_clients
        if not client.get("readonly") and client.get("directory")
    ]
    if len(candidates) == 1:
        return os.path.expanduser(candidates[0])
    return None


def _inode_signature(path: str) -> tuple[int, int] | None:
    try:
        stat_result = os.stat(path)
    except OSError:
        return None
    return (stat_result.st_dev, stat_result.st_ino)


def _get_buffer_content(
    checksum: Checksum, *, source_directory: str | None = None
) -> bytes | None:
    if source_directory is not None:
        path = _resolve_destination_path(
            source_directory, Checksum(checksum).hex(), create_dirs=False
        )
        if os.path.lexists(path):
            try:
                with open(path, "rb") as f:
                    buffer = f.read()
                if Buffer(buffer).get_checksum() == Checksum(checksum):
                    return buffer
            except Exception:
                pass
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
        "--destination",
        help="Read buffers directly from this destination buffer directory and bypass seamless-config",
    )

    transfer_mode_group = parser.add_mutually_exclusive_group()
    transfer_mode_group.add_argument(
        "--copy",
        help="Copy files from the destination buffer directory into the output paths",
        action="store_true",
    )
    transfer_mode_group.add_argument(
        "--hardlink",
        help="Create hardlinks from the destination buffer directory into the output paths",
        action="store_true",
    )
    transfer_mode_group.add_argument(
        "--softlink",
        "--symlink",
        dest="softlink",
        help="Create symlinks from the destination buffer directory into the output paths",
        action="store_true",
    )

    destination_policy_group = parser.add_mutually_exclusive_group()
    destination_policy_group.add_argument(
        "--dest-skip",
        dest="dest_skip",
        help="If the destination file already exists, trust it and skip verification",
        action="store_true",
    )
    destination_policy_group.add_argument(
        "--dest-verify",
        dest="dest_verify",
        help="If the destination file already exists, verify its contents and fail on mismatch",
        action="store_true",
    )
    destination_policy_group.add_argument(
        "--dest-repair",
        dest="dest_repair",
        help="If the destination file already exists and mismatches, replace it",
        action="store_true",
    )

    parser.add_argument(
        "--dest-rm",
        help="Remove existing extra files in target directories that are not listed in the downloaded index",
        action="store_true",
        default=False,
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
        "--incremental",
        help="Skip downloads whose target already shares an inode with the source buffer entry. Requires --hardlink.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compression",
        choices=["zst", "gz"],
        help="Append this compression suffix to all downloaded output files.",
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

    requested_transfer_mode = _get_transfer_mode(args)
    existing_entry_policy = _get_existing_entry_policy(args)
    source_directory = None
    transfer_mode = requested_transfer_mode

    if args.incremental and not args.hardlink:
        err("--incremental requires --hardlink")

    if args.destination:
        if args.project:
            err("--destination cannot be combined with --project")
        if args.stage:
            err("--destination cannot be combined with --stage")
        source_directory = os.path.expanduser(args.destination)
        if transfer_mode is None:
            transfer_mode = "copy"
    else:
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

    if source_directory is None:
        from seamless_config.extern_clients import set_remote_clients_from_env

        if not set_remote_clients_from_env(include_dask=False):
            seamless_config.init(workdir=os.getcwd())
            from seamless_config.select import get_selected_cluster

            selected_cluster = get_selected_cluster()
            if selected_cluster is None:
                print("Cannot download without a cluster defined", file=sys.stderr)
                return 1
        else:
            from seamless_config.select import get_selected_cluster

            selected_cluster = get_selected_cluster()

        is_local_cluster = False
        if selected_cluster is not None:
            try:
                from seamless_config.cluster import get_cluster

                is_local_cluster = get_cluster(selected_cluster).type == "local"
            except Exception:
                is_local_cluster = False

        if is_local_cluster:
            source_directory = _get_auto_destination_folder()
            if source_directory is not None and transfer_mode is None:
                transfer_mode = "copy"

        if requested_transfer_mode is not None and source_directory is None:
            if not is_local_cluster:
                err(
                    "Direct destination mode requires a selected local cluster or --destination"
                )
            err(f"--{requested_transfer_mode} requires a destination folder")

    ################################################################

    to_download = {}
    directories = []
    files = []
    index_checksums = {}
    compression_suffix = f".{args.compression}" if args.compression else ""
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
                index_buffer_content = _get_buffer_content(
                    index_checksum, source_directory=source_directory
                )
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
                    kk = os.path.join(dirname, k) + compression_suffix
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
        if compression_suffix:
            path += compression_suffix

        to_download[path] = checksum.hex()
        files.append(path)

    skip_targets = set()
    if args.incremental:
        if source_directory is None:
            err("--incremental requires a source destination folder")
        for target, checksum in to_download.items():
            _target_base, output_suffix = strip_compression_suffix(target)
            source_path = _resolve_destination_path(
                source_directory,
                checksum,
                create_dirs=False,
                compression_suffix=output_suffix,
            )
            if _inode_signature(target) == _inode_signature(source_path):
                skip_targets.add(target)

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
                extra_files = [f for f in existing_files if f not in to_download]
                if extra_files and not args.dest_rm:
                    err(
                        f"Directory '{directory}' contains {len(extra_files)} unlisted entries; use --dest-rm to remove them"
                    )
                for f in extra_files:
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
            path_base, _path_suffix = strip_compression_suffix(path)
            with open(path_base + ".CHECKSUM", "w") as f:
                f.write(checksum + "\n")
    elif args.stdout:
        if len(directories):
            err("Cannot download and print directory to stdout")
        if len(files) > 1:
            err("Cannot download and print multiple files to stdout")
        else:
            cs = to_download[files[0]]
            file_buffer = _get_buffer_content(
                Checksum(cs), source_directory=source_directory
            )
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
            source_directory=source_directory,
            transfer_mode=transfer_mode or "copy",
            existing_entry_policy=existing_entry_policy,
            skip_targets=skip_targets,
        )


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    finally:
        seamless.close()


if __name__ == "__main__":
    sys.exit(main())
