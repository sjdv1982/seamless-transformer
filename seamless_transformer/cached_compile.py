"""Compile Python code with built-in compile() and store it in linecache."""

from __future__ import annotations

import functools
import linecache
from ast import PyCF_ONLY_AST, Expr, FunctionDef, Lambda, stmt as Statement

from seamless.util.ipython import execute as ipython_execute


def cached_compile(code, identifier, mode="exec", flags=None, dont_inherit=0):
    """Compile Python code with built-in compile() and store it in linecache."""
    if flags is not None:
        astree = compile(code, identifier, mode, flags, dont_inherit)
    else:
        astree = compile(code, identifier, mode, dont_inherit=dont_inherit)
    cache_entry = (
        len(code),
        None,
        [line + "\n" for line in code.splitlines()],
        identifier,
    )
    linecache.cache[identifier] = cache_entry
    return astree


@functools.lru_cache(maxsize=1000)
def analyze_code(code, identifier):
    """Determine if Python code is a block, function, expression or lambda."""
    astree = cached_compile(code, identifier, "exec", PyCF_ONLY_AST)
    mode = "block"
    func_name = None
    if len(astree.body) == 1:
        if isinstance(astree.body[0], FunctionDef):
            mode = "function"
            func_name = astree.body[0].name
        elif isinstance(astree.body[0], Expr):
            if len(code.splitlines()) == 1:
                if isinstance(astree.body[0].value, Lambda):
                    mode = "lambda"
                else:
                    mode = "expression"
            else:
                err = (
                    "no multiline expressions or lambdas, "
                    "they may give indentation syntax errors"
                )
                raise SyntaxError((identifier, err))
    return mode, func_name


def exec_code(
    code, identifier, namespace, inputs, output, *, with_ipython_kernel=False
):
    """Execute Python code in a namespace."""
    mode, func_name = analyze_code(code, identifier)
    inputs2 = [inp for inp in sorted(list(inputs)) if not inp.endswith("_SCHEMA")]
    input_params = ",".join(f"{inp}={inp}" for inp in inputs2)
    if mode == "block":
        code2 = code
    elif mode == "function":
        code2 = code + "\n\n"
        if output is None:
            code2 += f"{func_name}({input_params})"
        else:
            code2 += f"{output} = {func_name}({input_params})"
    elif mode == "lambda":
        assert output is not None
        code2 = "LAMBDA = " + code + "\n\n"
        code2 += f"{output} = LAMBDA({input_params})"
    elif mode == "expression":
        assert output is not None
        code2 = f"{output} = " + code

    if with_ipython_kernel:
        ipython_execute(code2, namespace)
    else:
        code_obj = cached_compile(code2, identifier)
        exec(code_obj, namespace)


def check_function_like(code, identifier):
    """Check if code consists of one function and some statements."""
    astree = cached_compile(code, identifier, "exec", PyCF_ONLY_AST)
    if len(astree.body) == 1:
        return False
    function_name = None
    nstatements = 0
    for statement in astree.body:
        if isinstance(statement, FunctionDef):
            if function_name is not None:
                return False
            function_name = statement.name
        elif isinstance(statement, Statement):
            nstatements += 1
    if function_name is None:
        return False
    return function_name, nstatements


__all__ = ["cached_compile", "analyze_code", "exec_code", "check_function_like"]
