from concurrent.futures import ThreadPoolExecutor
import json
import sys
import os
import shutil

import seamless
import seamless.config as seamless_config
from seamless_transformer.cmd.message import (
    set_header,
    set_verbosity,
    message as msg,
    message_and_exit as err,
)
from seamless_transformer.cmd.file_load import files_to_checksums
from seamless_transformer.cmd.bytes2human import human2bytes
from seamless_transformer.cmd.exceptions import SeamlessSystemExit
from seamless import Checksum

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


def _main(argv: list[str] | None = None) -> int:

    import argparse

    parser = argparse.ArgumentParser(
        prog="seamless-upload",
        description="Upload files to a remote buffer server or folder",
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
        "-q", dest="verbosity", help="Quiet mode", action="store_const", const=-1
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
        "-m",
        "-mv",
        "--move",
        dest="move",
        help="""After successful upload, delete the original files and directories""",
        action="store_true",
    )

    parser.add_argument(
        "--hardlink",
        help="""Create hardlinks in the destination directory.
        By default, files and directories get copied into the destination directory.
        With this option, hardlinks are created instead.

        WARNING: never modify the original file in-place!!!
        Examples of in-place modification: 
            - "rsync --in-place",
            - appending to a file using ">>".
        This will lead to checksum corruption!!
        """,
        action="store_true",
    )

    parser.add_argument(
        "-y",
        "--yes",
        dest="auto_confirm",
        help="""Sets any confirmation values to 'yes' automatically. Users will not be asked to confirm any file upload or download.
        Uploads will happen without confirmation for up to 400 files and up to 100 MB in total.
        Downloads will happen without confirmation for up to 2000 files and up to 500 MB in total.
        These thresholds can be controlled by the environment variables:
        SEAMLESS_MAX_UPLOAD_FILES, SEAMLESS_MAX_UPLOAD_SIZE, SEAMLESS_MAX_DOWNLOAD_FILES, SEAMLESS_MAX_DOWNLOAD_SIZE.""",
        action="store_const",
        const="yes",
    )

    parser.add_argument(
        "-n",
        "--no",
        dest="auto_confirm",
        help="""Sets any confirmation values to 'no' automatically. Users will not be asked to confirm any file upload or download.
        Uploads will happen without confirmation for up to 400 files and up to 100 MB in total.
        Downloads will happen without confirmation for up to 2000 files and up to 500 MB in total.
        These thresholds can be controlled by the environment variables:
        SEAMLESS_MAX_UPLOAD_FILES, SEAMLESS_MAX_UPLOAD_SIZE, SEAMLESS_MAX_DOWNLOAD_FILES, SEAMLESS_MAX_DOWNLOAD_SIZE.""",
        action="store_const",
        const="no",
    )

    parser.add_argument("files_and_directories", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    for a in args.files_and_directories:
        if a.startswith("-"):
            err(f"Option {a} must be specified before upload targets")

    set_header("seamless-upload")
    verbosity = min(args.verbosity, 3)
    set_verbosity(verbosity)
    msg(1, "Verbosity set to {}".format(verbosity))

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

    max_upload_files = os.environ.get("SEAMLESS_MAX_UPLOAD_FILES", "400")
    max_upload_files = int(max_upload_files)
    max_upload_size = os.environ.get("SEAMLESS_MAX_UPLOAD_SIZE", "100 MB")
    max_upload_size = human2bytes(max_upload_size)

    from seamless_config.extern_clients import set_remote_clients_from_env

    if not set_remote_clients_from_env(include_dask=False):

        seamless_config.init(workdir=os.getcwd())
        from seamless_config.select import get_selected_cluster

        if get_selected_cluster() is None:
            print(f"Cannot upload without a cluster defined", file=sys.stderr)
            exit(1)

    destination_folder = None
    try:
        import seamless_remote.buffer_remote as buffer_remote
    except Exception:
        buffer_clients = []
    else:
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
        destination_folder = os.path.expanduser(candidates[0])
        if args.auto_confirm is None:
            args.auto_confirm = "yes"
        msg(1, f"Use buffer directory '{destination_folder}' for upload")

    if args.hardlink and not destination_folder:
        err("--hardlink requires a destination folder")

    paths = [
        path.rstrip(os.sep)
        for path in args.files_and_directories
        if not path.endswith(".CHECKSUM")
    ]
    directories = [path for path in paths if os.path.isdir(path)]

    try:
        file_checksum_dict, directory_indices = files_to_checksums(
            paths,
            max_upload_size=max_upload_size,
            max_upload_files=max_upload_files,
            directories=directories,
            auto_confirm=args.auto_confirm,
            destination_folder=destination_folder,
            hardlink_destination=args.hardlink,
        )
    except SeamlessSystemExit as exc:
        err(*exc.args)

    def write_checksum(filename, file_checksum):
        if file_checksum is None:
            return
        file_checksum = Checksum(file_checksum)
        try:
            filename2 = filename
            if filename.endswith(".INDEX"):
                filename2 = os.path.splitext(filename)[0]
            with open(filename2 + ".CHECKSUM", "w") as f:
                f.write(file_checksum.hex() + "\n")
        except Exception:
            msg(0, f"Cannot write checksum to file '{filename}.CHECKSUM'")
            return

    for dirname in directories:
        index_buffer, index_checksum = directory_indices[dirname]
        try:
            with open(dirname + ".INDEX", "wb") as f:
                f.write(index_buffer)
        except Exception:
            msg(0, f"Cannot write directory index to file '{dirname}.INDEX'")
        else:
            write_checksum(dirname, index_checksum)

    filenames = [path for path in paths if path not in directories]
    with ThreadPoolExecutor(max_workers=100) as executor:
        executor.map(
            write_checksum, filenames, [file_checksum_dict[path] for path in filenames]
        )

    if args.move:
        with ThreadPoolExecutor(max_workers=100) as executor:
            executor.map(shutil.rmtree, paths)


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    finally:
        seamless.close()


if __name__ == "__main__":
    sys.exit(main())
