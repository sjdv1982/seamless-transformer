"""Utils related to Python source code"""

import ast
import inspect
import os
import textwrap
from types import LambdaType


def strip_decorators(code: str) -> str:
    """Remove decorators from a Python source code string"""
    lines = code.splitlines()
    lnr = 0
    for lnr, line in enumerate(lines):
        if not line.startswith("@"):
            break
    return "\n".join(lines[lnr:])


def ast_dump(node, annotate_fields=True, include_attributes=False, *, indent=None):
    """
    From the CPython 3.10 source code, https://github.com/python/cpython/blob/3.10/Lib/ast.py
    :copyright: Copyright 2008 by Armin Ronacher.
    :license: Python License.

    Return a formatted dump of the tree in node.  This is mainly useful for
    debugging purposes.  If annotate_fields is true (by default),
    the returned string will show the names and the values for fields.
    If annotate_fields is false, the result string will be more compact by
    omitting unambiguous field names.  Attributes such as line
    numbers and column offsets are not dumped by default.  If this is wanted,
    include_attributes can be set to true.  If indent is a non-negative
    integer or string, then the tree will be pretty-printed with that indent
    level. None (the default) selects the single line representation.
    """
    from ast import AST

    def _format(node, level=0):
        if indent is not None:
            level += 1
            prefix = "\n" + indent * level
            sep = ",\n" + indent * level
        else:
            prefix = ""
            sep = ", "
        if isinstance(node, AST):
            cls = type(node)
            args = []
            allsimple = True
            keywords = annotate_fields
            for name in node._fields:
                try:
                    value = getattr(node, name)
                except AttributeError:
                    keywords = True
                    continue
                if value is None and getattr(cls, name, ...) is None:
                    keywords = True
                    continue
                value, simple = _format(value, level)
                allsimple = allsimple and simple
                if keywords:
                    args.append("%s=%s" % (name, value))
                else:
                    args.append(value)
            if include_attributes and node._attributes:
                for name in node._attributes:
                    try:
                        value = getattr(node, name)
                    except AttributeError:
                        continue
                    if value is None and getattr(cls, name, ...) is None:
                        continue
                    value, simple = _format(value, level)
                    allsimple = allsimple and simple
                    args.append("%s=%s" % (name, value))
            if allsimple and len(args) <= 3:
                return "%s(%s)" % (node.__class__.__name__, ", ".join(args)), not args
            return "%s(%s%s)" % (node.__class__.__name__, prefix, sep.join(args)), False
        elif isinstance(node, list):
            if not node:
                return "[]", True
            return (
                "[%s%s]" % (prefix, sep.join(_format(x, level)[0] for x in node)),
                False,
            )
        return repr(node), True

    if not isinstance(node, AST):
        raise TypeError("expected AST, got %r" % node.__class__.__name__)
    if indent is not None and not isinstance(indent, str):
        indent = " " * indent
    return _format(node)[0]


def lambdacode(lambda_func):
    """Return the source of a (short) lambda function.
    If it's impossible to obtain, returns None.
    """
    try:
        source_lines, _ = inspect.getsourcelines(lambda_func)
    except (IOError, TypeError):
        return None

    # skip `def`-ed functions and long lambdas
    if len(source_lines) != 1:
        return None

    source_text = os.linesep.join(source_lines).strip()

    # find the AST node of a lambda definition
    # so we can locate it in the source code
    source_ast = ast.parse(source_text)
    lambda_node = next(
        (node for node in ast.walk(source_ast) if isinstance(node, ast.Lambda)), None
    )
    if lambda_node is None:  # could be a single line `def fn(x): ...`
        return None

    # HACK: Since we can (and most likely will) get source lines
    # where lambdas are just a part of bigger expressions, they will have
    # some trailing junk after their definition.
    #
    # Unfortunately, AST nodes only keep their _starting_ offsets
    # from the original source, so we have to determine the end ourselves.
    # We do that by gradually shaving extra junk from after the definition.
    lambda_text = source_text[lambda_node.col_offset :]
    lambda_body_text = source_text[lambda_node.body.col_offset :]
    min_length = len("lambda:_")  # shortest possible lambda expression
    while len(lambda_text) > min_length:
        try:
            # What's annoying is that sometimes the junk even parses,
            # but results in a *different* lambda. You'd probably have to
            # be deliberately malicious to exploit it but here's one way:
            #
            #     bloop = lambda x: False, lambda x: True
            #     get_short_lamnda_source(bloop[0])
            #
            # Ideally, we'd just keep shaving until we get the same code,
            # but that most likely won't happen because we can't replicate
            # the exact closure environment.
            code = compile(lambda_body_text, "<unused filename>", "eval")

            # Thus the next best thing is to assume some divergence due
            # to e.g. LOAD_GLOBAL in original code being LOAD_FAST in
            # the one compiled above, or vice versa.
            # But the resulting code should at least be the same *length*
            # if otherwise the same operations are performed in it.
            if len(code.co_code) == len(lambda_func.__code__.co_code):
                return lambda_text
        except SyntaxError:
            pass
        lambda_text = lambda_text[:-1]
        lambda_body_text = lambda_body_text[:-1]

    return None


def getsource(func):
    """Get the source of a Python function"""

    if isinstance(func, LambdaType) and func.__name__ == "<lambda>":
        code = lambdacode(func)
        if code is None:
            raise ValueError("Cannot extract source code from this lambda")
        return code
    code = inspect.getsource(func)
    code = textwrap.dedent(code)
    code = strip_decorators(code)
    return code
