"""Command-line argument parsing for command-line seamless.
Command-line arguments can be files, directories, constants, or bash code."""

import time
from typing import Any
from pathlib import Path
import os
from collections import namedtuple

import bashlex

from seamless.checksum.calculate_checksum import calculate_file_checksum
from .message import message as msg, message_and_exit as err
from .file_load import read_checksum_file


def _has_explicit_sidecar_suffix(arg: str) -> bool:
    return arg.endswith(".CHECKSUM") or arg.endswith(".INDEX")


def fill_checksum_arguments(file_args: list, order: list[str]):
    for n, file_arg in enumerate(file_args.copy()):
        if isinstance(file_arg, str):
            newf = {"name": file_arg, "mapping": file_arg}
        else:
            newf = file_arg

        arg = newf["mapping"]
        if _has_explicit_sidecar_suffix(arg):
            continue
        change = False
        if arg.endswith(".CHECKSUM"):
            arg0 = os.path.splitext(arg)[0]
            arg_cs = arg
            newf["mapping"] = arg0
            change = True
        else:
            arg0 = arg
            arg_cs = arg + ".CHECKSUM"

        if os.path.exists(arg_cs):
            checksum = read_checksum_file(arg_cs)
            newf["checksum"] = checksum
            change = True
        if change:
            file_args[n] = newf
            try:
                pos = order.index(file_arg)
                order[pos] = arg0
            except ValueError:
                pass


def guess_arguments_with_custom_error_messages(
    args: list[str],
    *,
    rule_ext_error_message,
    rule_no_ext_error_message,
    rule_no_slash_error_message,
    overrule_ext: bool = False,
    overrule_no_ext: bool = False,
) -> dict[str, Any]:
    """Guess for each argument if it represents a file, directory or value.

    In principle, for each argument,
     if it exists as a file/directory, it will be a file/directory, else a value.

    But there are three rules that must be respected, else an exception is raised.

    1. Any argument with extension must exist as a file, but not as a directory.
       However, numeric arguments such as "0.1" are always interpreted as value.
    2. Any argument (beyond the first) without extension must not exist as a file
       (directories are fine)
    3. Any argument ending with a slash must be a directory

    Input:
    - args: list of arguments

    - rule_ext_error_message: If rule 1. is violated, the ValueError message to raise.
    The message will be prepended with "Argument does not exist"
     or "Argument is a directory".
    It will then be formatted with "argindex" as the argument index (counting from 1)
    and "arg" as the argument string.

    - rule_no_ext_error_message: If rule 2. is violated, the ValueError message to raise.
    The message will be formatted with "argindex" as the argument index (counting from 1)
    and "arg" as the argument string.

    - rule_no_slash_error_message: If rule 3. is violated, the ValueError message to raise.
    The message will be prepended with "Argument does not exist"
     or "Argument is not a directory".
    It will then be formatted with "argindex" as the argument index (counting from 1)
    and "arg" as the argument string.

    - overrule_ext: if True, rule 1. does not apply.

    - overrule_no_ext: if True, rule 2. does not apply.

    Output:
    dict of argname -> mode
    where mode is "file", "directory" or "value"
    """

    result = {"@order": args}
    for argindex0, arg in enumerate(args.copy()):
        argindex = argindex0 + 1
        path = Path(arg)
        future_path = Path(arg + ".FUTURE")
        if future_path.exists():
            msg(0, f"Waiting for future '{future_path}'...")
            while 1:
                age = time.time() - future_path.stat().st_atime
                if age > 120:
                    err("Stale future {future_path}")
                time.sleep(0.5)
                if not future_path.exists():
                    break

        checksum_path = Path(arg + ".CHECKSUM")
        index_path = Path(arg + ".INDEX")
        checksum = None
        if checksum_path.exists():
            item = {}
            if index_path.exists():
                item["type"] = "directory"
                msg(
                    3,
                    # pylint: disable=line-too-long
                    "Argument #{} '{}', .CHECKSUM and .INDEX file exist, read directory checksum".format(
                        argindex, arg
                    ),
                )
            else:
                item["type"] = "file"
                msg(
                    3,
                    "Argument #{} '{}', .CHECKSUM file exists, read file checksum".format(
                        argindex, arg
                    ),
                )
            checksum = read_checksum_file(checksum_path.as_posix())
            if not checksum:
                err(
                    # pylint: disable=line-too-long
                    "Argument #{} '{}', .CHECKSUM file exists, but does not contain a valid checksum".format(
                        argindex, arg
                    )
                )
            item["checksum"] = checksum
            result[arg] = item

        extension = path.suffix
        msg(3, "Argument #{} '{}', extension: '{}'".format(argindex, arg, extension))
        exists = path.exists() or path.expanduser().exists()
        is_dir = False
        if exists:
            is_dir = path.is_dir() or path.expanduser().is_dir()
        is_float = False
        if extension:
            try:
                float(arg)
                is_float = True
            except ValueError:
                pass

        if checksum:
            if exists and not is_dir:
                arg2 = os.path.expanduser(arg)
                file_checksum = calculate_file_checksum(arg2)
                if file_checksum != checksum:
                    raise ValueError(
                        # pylint: disable=line-too-long
                        "Argument exists as file and .CHECKSUM file, but the checksums are not the same"
                    )
            continue

        # Rule 1.: Any argument with extension must exist as a file, but not as a directory.
        # However, numeric arguments such as "0.1" are always interpreted as numeric.

        if not overrule_ext:
            if extension and not is_float and not arg.endswith(os.sep):
                if not exists:
                    errmsg = "Argument does not exist.\n" + rule_ext_error_message
                    raise ValueError(errmsg.format(argindex=argindex, arg=arg))
                if is_dir:
                    errmsg = "Argument is a directory.\n" + rule_ext_error_message
                    raise ValueError(errmsg.format(argindex=argindex, arg=arg))

        # Rule 2.: Any argument (beyond the first) without extension must not exist as a file
        #          (directories are fine)
        if not overrule_no_ext:
            if argindex > 1 and not extension:
                if exists and not is_dir:
                    errmsg = rule_no_ext_error_message
                    raise ValueError(errmsg.format(argindex=argindex, arg=arg))

        # Rule 3.: Any argument ending with a slash must be a directory
        if arg.endswith(os.sep):
            if not exists:
                errmsg = "Argument does not exist.\n" + rule_no_slash_error_message
                raise ValueError(errmsg.format(argindex=argindex, arg=arg))
            if not is_dir:
                errmsg = "Argument is not a directory.\n" + rule_no_slash_error_message
                raise ValueError(errmsg.format(argindex=argindex, arg=arg))

        if is_float:
            item = "value"
        elif exists:
            if is_dir:
                result_mode = "directory"
                arg2 = os.path.expanduser(arg)
            else:
                result_mode = "file"
                arg2 = os.path.expanduser(arg)
            item = {"type": result_mode}
            if arg2 != arg:
                item["mapping"] = arg2
        else:
            item = "value"
        result[arg] = item

    return result


def guess_arguments(
    args: list[str],
    *,
    overrule_ext: bool = False,
    overrule_no_ext: bool = False,
) -> dict[str, Any]:
    """Guess for each argument if it represents a file, directory or value.

    In principle, for each argument,
     if it exists as a file/directory, it will be a file/directory, else a value.

    But there are three rules that must be respected, else an exception is raised.

    1. Any argument with extension must exist as a file, but not as a directory.
       However, numeric arguments such as "0.1" are always interpreted as value.
    2. Any argument (beyond the first) without extension must not exist as a file
       (directories are fine)
    3. Any argument ending with a slash must be a directory

    Special case: if argument.CHECKSUM exists, the checksum is read directly
    from argument.CHECKSUM. In that case, if argument.INDEX exists as well,
    (regardless of its contents). the argument is considered as a directory,
    else as a file.

    If an argument explicitly ends in .CHECKSUM or .INDEX, it is treated as a
    literal file path and is not dereferenced to the base path.

    Input:
    - args: list of arguments

    - overrule_ext: if True, rule 1. does not apply.

    - overrule_no_ext: if True, rule 2. does not apply.
    """

    rule_ext_error_message = """Argument #{argindex} '{arg}' has an extension.
Therefore, it must exist as a file.
To disable this rule, specify the -g1 option."""
    # TODO: add something in case of -c and ?/*
    rule_no_ext_error_message = """Argument #{argindex} '{arg}' has no extension.
Unless it is the first argument, it can't exist as a file.
To disable this rule, specify the -g2 option."""
    rule_no_slash_error_message = """Argument #{argindex} '{arg}' ends with a slash.
Therefore, it must be a directory."""

    return guess_arguments_with_custom_error_messages(
        args,
        overrule_ext=overrule_ext,
        overrule_no_ext=overrule_no_ext,
        rule_ext_error_message=rule_ext_error_message,
        rule_no_ext_error_message=rule_no_ext_error_message,
        rule_no_slash_error_message=rule_no_slash_error_message,
    )


Command = namedtuple(
    "Command", ("start", "end", "main_node", "wordnodes", "words", "commandstring")
)


class _WordVisitor(bashlex.ast.nodevisitor):
    def __init__(self):
        self.words = []
        self.nodes = []
        # barrier *should* be redundant, but you never know.
        # The words must be the correct ones for the interface .py file to get the correct arguments
        self.barrier = None
        super().__init__()

    def visitword(self, n, _):
        node = n
        self.nodes.append(node)
        self.words.append(node.word)
        return True

    def visitredirect(
        self, node, *args, **kwargs
    ):  # pylint: disable = arguments-differ, unused-argument
        start = node.pos[0]
        if self.barrier is None or self.barrier < start:
            self.barrier = start
        return False

    def _filter(self):
        if self.barrier is None:
            return
        self.nodes[:] = [node for node in self.nodes if node.pos[1] < self.barrier]
        self.words[:] = [node.word for node in self.nodes]


class _CommandVisitor(bashlex.ast.nodevisitor):
    def __init__(self, full_commandstring):
        self.commands = []
        self.full_commandstring = full_commandstring
        super().__init__()

    def visitcommand(self, n, _):
        node = n
        wordvisitor = _WordVisitor()
        wordvisitor.visit(node)
        wordvisitor._filter()
        start, end = node.pos
        cmd = Command(
            main_node=node,
            start=start,
            end=end,
            wordnodes=wordvisitor.nodes,
            words=wordvisitor.words,
            commandstring=self.full_commandstring[start:end],
        )
        self.commands.append(cmd)
        return True


def get_primary_pipeline(
    bashtrees: list, commands: "list[Command]", primary_index: int
) -> tuple[int, int]:
    """Find the pipeline (or command node) containing commands[primary_index].

    Returns (start_idx, end_idx) as indices into commands, where
    commands[start_idx:end_idx] are all commands in the same pipeline.

    Same logic as get_commands for primary_index=0, but position-driven
    instead of always taking the first element.
    """
    target_start = commands[primary_index].start

    # Start from whichever top-level tree contains the target command.
    first = bashtrees[0]
    for bashtree in bashtrees:
        if bashtree.pos[0] <= target_start < bashtree.pos[1]:
            first = bashtree
            break

    # Peel one layer of compound/list — same as get_commands, but find the
    # element containing target_start rather than always taking index 0.
    if first.kind == "compound" and first.list:
        for child in first.list:
            if hasattr(child, "pos") and child.pos[0] <= target_start < child.pos[1]:
                first = child
                break
    elif first.kind == "list" and first.parts:
        for part in first.parts:
            if hasattr(part, "pos") and part.pos[0] <= target_start < part.pos[1]:
                first = part
                break

    if first.kind == "command":
        return (primary_index, primary_index + 1)
    if first.kind == "pipeline":
        pipe_start, pipe_end = first.pos
        start_idx = end_idx = None
        for i, com in enumerate(commands):
            if com.start >= pipe_start and com.end <= pipe_end:
                if start_idx is None:
                    start_idx = i
                end_idx = i + 1
        return (start_idx, end_idx)
    return (primary_index, primary_index + 1)


def get_commands(
    commandstring: str,
    primary: int = 0,
) -> tuple[list[Command], tuple[int, int] | None, bashlex.parser.ast.node | None]:
    """Parse a bash command string into a list of Command instances.
    The range of the primary bash pipeline is also returned as (start, end) indices.
    If the command is a pipeline between parentheses, its redirect is returned as well.
    """
    try:
        bashtrees = bashlex.parse(commandstring)
    except Exception:
        raise ValueError("Unrecognized bash syntax") from None
    visitor = _CommandVisitor(commandstring)
    for bashtree in bashtrees:
        visitor.visit(bashtree)
    commands = sorted(visitor.commands, key=lambda command: command.start)
    primary_pipeline = None
    pipeline_redirect = None
    if len(bashtrees):
        first = bashtrees[0]
        if first.kind == "compound" and len(first.list):
            v = _RedirectionVisitor()
            v.visit(first)
            redirect = v.redirect
            if redirect is None:
                redirect = v.maybe_redirect
            if redirect is not None:
                pipeline_redirect = redirect.output
            first = first.list[0]
        elif first.kind == "list" and len(first.parts):
            first = first.parts[0]

        if commands:
            primary_pipeline = get_primary_pipeline(bashtrees, commands, primary)
    return commands, primary_pipeline, pipeline_redirect


class _RedirectionVisitor(bashlex.ast.nodevisitor):
    def __init__(self):
        self.redirect = None
        self.maybe_redirect = None
        super().__init__()

    def visitredirect(self, node, *args, **kwargs):
        # pylint: disable = arguments-differ, unused-argument
        maybe = False
        if node.output.word.startswith("<"):
            return
        if isinstance(node.input, int) and node.input == 2:
            return
        if isinstance(node.input, int) and node.input != 1:
            maybe = True
        if maybe:
            self.maybe_redirect = node
        else:
            if self.redirect is not None:
                msg(-1, "Multiple redirects in the last command")
                exit(1)
            self.redirect = node


def get_redirection(command: Command):
    """Return the redirection output of a bash command"""
    visitor = _RedirectionVisitor()
    visitor.visit(command.main_node)
    redirect = visitor.redirect
    if redirect is None:
        redirect = visitor.maybe_redirect
    if redirect is None:
        return None
    return redirect.output
