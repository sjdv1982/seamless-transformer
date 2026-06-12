"""Executable path resolution for cmd-seamless"""

import os
import subprocess
from pathlib import Path

from .message import message as msg


def resolve_executable(command):
    """Resolve the mapped executable path for the first command word.

    Returns the mapped executable path, or None if the first arg is a POSIX tool.
    """
    mapped_execarg = None

    args1 = [Path(command[0]), Path(command[0]).expanduser()]
    for arg1 in args1:
        if arg1.as_posix().strip() in ("conda",):
            is_posix = True
        else:
            execarg1 = subprocess.getoutput("which {}".format(arg1.as_posix())).strip()
            if execarg1:
                msg(
                    2,
                    "first argument '{}' is in PATH, map to '{}'".format(
                        arg1.as_posix(), execarg1
                    ),
                )
                execarg1dir = os.path.split(execarg1)[0]
                if (
                    not execarg1dir.endswith("/bin")
                    and not execarg1dir.endswith("/sbin")
                    and not execarg1dir.endswith("/usr")
                ):
                    msg(
                        1,
                        "first argument '{}' does not seem a POSIX tool. Explicitly upload it as '{}'".format(  # pylint: disable=line-too-long
                            arg1.as_posix(), execarg1
                        ),
                    )
                    mapped_execarg = execarg1

                arg1 = Path(execarg1)
                is_posix = False
            else:
                is_posix = True
        if is_posix:
            mapped_execarg = arg1.as_posix()
        break

    return mapped_execarg
