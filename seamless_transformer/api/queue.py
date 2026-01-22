"""Seamless queue server. Reads commands from SEAMLESS_QUEUE_FILE (default: .seamless-queue)"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import os
import sys
import threading
import time

import seamless
import seamless.config as seamless_config

from seamless_transformer.cmd.bash_transformation import run_transformation_async
from seamless_transformer.cmd.get_results import (
    get_result_buffer_async,
    get_results,
    maintain_futures,
)

__version__ = "0.14"


def _print_message(*args, prefix: str | None = None, label: str | None = None) -> None:
    if args and isinstance(args[0], int):
        args = args[1:]
    if label:
        args = (label, *args)
    if prefix:
        args = (prefix, *args)
    print(*args, file=sys.stderr)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-q", "--quiet", dest="quiet", help="Quiet mode", action="store_true"
    )
    return parser.parse_args(argv)


_job_count = 0


async def _process_job(command: dict, *, quiet: bool) -> None:
    global _job_count
    _job_count += 1
    job_count = _job_count
    original_command = command["original_command"]
    try:
        transformation_checksum = command["transformation_checksum"]
        transformation_dict = command["transformation_dict"]
        result_targets = command["result_targets"]
        params = command["params"]
        undo = params["undo"]
        scratch = params["scratch"]
        workdir = params["workdir"]
        auto_confirm = params["auto_confirm"]
        assert auto_confirm in ("yes", "no")
        max_download_size = params["max_download_size"]
        max_download_files = params["max_download_files"]
        capture_stdout = params["capture_stdout"]
        download = params["download"]
    except KeyError:
        _print_message(
            f"""
*** Job {_job_count}: {original_command} ***
Malformed command
"""
        )
        return

    if not quiet:
        _print_message(f"Job {job_count}, run command:")
        _print_message(original_command)
        _print_message()

    delete_futures_event = threading.Event()

    try:
        header = f"*** Job {job_count}: {original_command} ***\n"
        maintain_fut = functools.partial(
            maintain_futures,
            workdir,
            transformation_checksum,
            result_targets,
            delete_futures_event=delete_futures_event,
            msg_func=functools.partial(_print_message, prefix=header),
        )
        maintain_thread = threading.Thread(target=maintain_fut, name="maintain_futures")
        maintain_thread.start()

        result_checksum = await run_transformation_async(
            transformation_dict, undo=undo, fingertip=True, scratch=scratch
        )
        result_buffer = await get_result_buffer_async(
            result_checksum,
            do_fingertip=True,
            do_scratch=scratch,
            has_result_targets=True,
            err_func=functools.partial(_print_message, prefix=header, label="ERROR:"),
        )
        get_results(
            result_targets,
            result_checksum,
            result_buffer,
            workdir=workdir,
            do_scratch=scratch,
            do_download=download,
            do_capture_stdout=capture_stdout,
            do_auto_confirm=auto_confirm,
            max_download_size=max_download_size,
            max_download_files=max_download_files,
            msg_func=functools.partial(_print_message, prefix=header),
        )
        if not quiet:
            _print_message(f"*** Job {job_count}: {original_command}, FINISHED ***")
    finally:
        delete_futures_event.set()


async def _queue_loop(queue_file: str, *, quiet: bool) -> bool:
    eof = False
    while not os.path.exists(queue_file):
        await asyncio.sleep(0.5)

    running_jobs: dict[str, asyncio.Future] = {}
    job_count = 0
    with open(queue_file, "rb") as q:
        fsize = 0
        while not eof:
            q.seek(0, 2)
            if q.tell() != fsize:
                q.seek(fsize)
                data = q.read()
                if data.endswith(b"\x00"):
                    commands = data.split(b"\x00")
                    for command in commands:
                        if not len(command):
                            continue
                        try:
                            command = command.decode()
                            command = json.loads(command)
                            queue_command = command["queue_command"]
                        except UnicodeDecodeError:
                            _print_message("Command is not valid text", label="ERROR:")
                            continue
                        except json.JSONDecodeError:
                            _print_message("Command is not valid JSON", label="ERROR:")
                            continue
                        except KeyError:
                            _print_message(
                                "Command is not a valid queue command", label="ERROR:"
                            )
                            continue

                        if queue_command == "EOF":
                            eof = True
                        elif queue_command == "SUBMIT":
                            original_command = command["original_command"]
                            job_count += 1
                            job_str = f"Job {job_count}: {original_command}"
                            future = asyncio.ensure_future(
                                _process_job(command, quiet=quiet)
                            )
                            running_jobs[job_str] = future

                fsize = q.tell()

            try:
                for job_str, job in list(running_jobs.items()):
                    if job.done():
                        running_jobs.pop(job_str)
                        if job.exception() is not None:
                            _print_message(
                                f"*** {job_str}, EXCEPTION ***\n" + str(job.exception())
                            )
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                if not quiet:
                    _print_message("CANCEL")
                break
    if eof:
        await asyncio.gather(*running_jobs.values())
        for job_str, job in running_jobs.items():
            if job.exception() is not None:
                _print_message(f"*** {job_str}, EXCEPTION ***\n" + str(job.exception()))
        return True
    return False


def _queue_file_from_env(quiet: bool) -> str:
    queue_file = os.environ.get("SEAMLESS_QUEUE_FILE")
    if queue_file is None:
        queue_file = ".seamless-queue"
        if not quiet:
            _print_message(
                f"SEAMLESS_QUEUE_FILE not defined. Set queue file to '{queue_file}'"
            )
    return queue_file


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    seamless_config.init(workdir=os.getcwd())

    queue_file = _queue_file_from_env(args.quiet)

    finished_all_jobs = False
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        finished_all_jobs = loop.run_until_complete(
            _queue_loop(queue_file, quiet=args.quiet)
        )
    except KeyboardInterrupt:
        pass
    finally:
        seamless.close()

    if finished_all_jobs:
        os.unlink(queue_file)

    logger = logging.getLogger("seamless")
    logger.setLevel(logging.ERROR)
    return 0


def finish(argv: list[str] | None = None) -> int:
    if argv:
        _parse_args(argv)

    queue_file = os.environ.get("SEAMLESS_QUEUE_FILE")
    if queue_file is None:
        queue_file = ".seamless-queue"

    exists = os.path.exists(queue_file)

    with open(queue_file, "ab") as fp:
        msg = json.dumps({"queue_command": "EOF"})
        fp.write(msg.encode() + b"\x00")

    if exists:
        while os.path.exists(queue_file):
            time.sleep(0.1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
