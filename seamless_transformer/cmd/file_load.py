"""Utilities to load files"""

from concurrent.futures import ThreadPoolExecutor
import functools
import os

from seamless.checksum.calculate_checksum import calculate_checksum
from seamless.checksum.serialize import serialize_sync as serialize
from seamless import Checksum
from .message import message as msg
from .register import register_file, register_buffer, check_checksums_present
from .bytes2human import bytes2human
from .confirm import confirm_yn
from .exceptions import SeamlessSystemExit


def strip_textdata(data):
    """Strip textdata"""
    while 1:
        old_len = len(data)
        data = data.strip("\n")
        data = data.strip()
        if len(data) == old_len:
            break
    lines = [l for l in data.splitlines() if not l.lstrip().startswith("#")]
    return "\n".join(lines)


def read_checksum_file(filename) -> str | None:
    """Read .CHECKSUM file"""
    with open(filename) as f:
        checksum = f.read()
    checksum = strip_textdata(checksum)
    try:
        return Checksum(checksum).hex()
    except Exception:
        return None


def _file_checksum_and_length(filename: str) -> tuple[str, int]:
    with open(filename, "rb") as f:
        buffer = f.read()
    return calculate_checksum(buffer), len(buffer)


def files_to_checksums(
    filelist: list[str],
    *,
    directories=dict[str, str],
    max_upload_files: int | None,
    max_upload_size: int | None,
    nparallel: int = 20,
    auto_confirm: str | None,
    destination_folder: str | None = None,
    hardlink_destination: bool = False,
    dry_run: bool = False
):
    """Convert a list of filenames to a dict of filename-to-checksum items
    In addition, each file buffer is uploaded.

    max_upload_files: the maximum number of files to send to the database.
    max_upload_size: the maximum data size (in bytes) to send to the database.
    nparallel: number of files to process simultaneously
    directories:
      keys are entries in filelist that are directories instead of files.
      values are the mapped directory paths (as they will appear on the server)
    destination_folder: instead of uploading to a buffer server, write to this folder
    hardlink_destination: in the destination folder, don't write files, create hardlinks instead.
    """

    all_filelist = [f for f in filelist if f not in directories]
    directory_files = {}
    for dirname in directories:
        directory_files[dirname] = []
        for dirpath, _, filenames in os.walk(dirname):
            assert dirpath.startswith(dirname), (dirpath, dirname)
            dirtail = dirpath[len(dirname) + 1 :]
            for filename in filenames:
                full_filename = os.path.join(dirpath, filename)
                mapped_filename = os.path.join(dirtail, filename)
                directory_files[dirname].append((full_filename, mapped_filename))
                all_filelist.append(full_filename)

    upload_buffer_lengths = {}
    all_result = {}
    with ThreadPoolExecutor(max_workers=nparallel) as executor:
        file_infos = list(executor.map(_file_checksum_and_length, all_filelist))

    file_checksums = [info[0] for info in file_infos]
    present_checksums = check_checksums_present(
        file_checksums, destination_folder=destination_folder
    )

    upload_filelist = []
    for filename, (checksum, buffer_length) in zip(all_filelist, file_infos):
        all_result[filename] = checksum
        has_buffer = checksum in present_checksums
        if not has_buffer:
            if checksum in upload_buffer_lengths:
                msg(
                    2,
                    "Duplicate file: '{}', checksum {}, length {}".format(
                        filename, checksum, buffer_length
                    ),
                )
            else:
                msg(
                    2,
                    "Not in remote storage: '{}', checksum {}, length {}".format(
                        filename, checksum, buffer_length
                    ),
                )
                upload_filelist.append(filename)
                upload_buffer_lengths[checksum] = buffer_length
        else:
            msg(
                2,
                "Already in remote storage: '{}', checksum {}, length {}".format(
                    filename, checksum, buffer_length
                ),
            )

    deepfolder_buffers = {}
    for dirname in directories:
        deepfolder = {d[1]: all_result[d[0]] for d in directory_files[dirname]}
        deepfolder_buffers[dirname] = serialize(deepfolder, "plain")

    directory_indices = {}
    upload_buffers = {}
    if directories:
        dir_checksums = {}
        for dirname, buffer in deepfolder_buffers.items():
            checksum = calculate_checksum(buffer)
            dir_checksums[dirname] = checksum
        present_dir_checksums = check_checksums_present(
            list(dir_checksums.values()), destination_folder=destination_folder
        )
        for dirname, checksum in dir_checksums.items():
            buffer = deepfolder_buffers[dirname]
            directory_indices[dirname] = buffer, checksum
            buffer_length = len(buffer)
            all_result[dirname] = checksum
            has_buffer = checksum in present_dir_checksums
            if not has_buffer:
                if checksum in upload_buffer_lengths:
                    msg(
                        2,
                        "Duplicate directory: Index of '{}', checksum {}, length {}".format(
                            dirname, checksum, buffer_length
                        ),
                    )
                else:
                    msg(
                        2,
                        "Not in remote storage: index of '{}', checksum {}, length {}".format(
                            dirname, checksum, buffer_length
                        ),
                    )
                    upload_buffers[checksum] = buffer
                    upload_buffer_lengths[checksum] = buffer_length
            else:
                msg(
                    2,
                    "Already in remote storage: index of '{}', checksum {}, length {}".format(
                        dirname, checksum, buffer_length
                    ),
                )

    datasize = sum(upload_buffer_lengths.values())
    size = bytes2human(datasize, format="%(value).2f %(symbol)s")
    ask_confirmation = False
    if max_upload_files is not None and len(upload_buffer_lengths) > max_upload_files:
        ask_confirmation = True
    elif max_upload_size is not None and datasize > max_upload_size:
        ask_confirmation = True
    if auto_confirm == "yes":
        ask_confirmation = False
    if dry_run:
        ask_confirmation = False
    if ask_confirmation:
        if auto_confirm == "no":
            err = "Cannot confirm upload of {} files, total {}. Exiting.".format(
                len(upload_buffer_lengths), size
            )
            raise SeamlessSystemExit(err)
        confirmation = confirm_yn(
            "Confirm upload of {} files, total {}?".format(
                len(upload_buffer_lengths), size
            ),
            default="no",
        )
        if not confirmation:
            raise SeamlessSystemExit("Exiting.")
    if len(upload_buffer_lengths):
        msg(0, "Upload {} files, total {}".format(len(upload_buffer_lengths), size))
    else:
        msg(1, "Upload no files")

    if upload_filelist and not dry_run:
        reg_file = functools.partial(
            register_file,
            destination_folder=destination_folder,
            hardlink=hardlink_destination,
        )
        with ThreadPoolExecutor(max_workers=nparallel) as executor:
            for filename, checksum in zip(
                upload_filelist, executor.map(reg_file, upload_filelist)
            ):
                assert all_result[filename] == checksum, (
                    filename,
                    checksum,
                    all_result[filename],
                )

    if upload_buffers and not dry_run:
        reg_buf = functools.partial(
            register_buffer, destination_folder=destination_folder
        )
        with ThreadPoolExecutor(max_workers=nparallel) as executor:
            executor.map(reg_buf, upload_buffers.values())

    result = {filename: all_result[filename] for filename in filelist}

    return result, directory_indices
