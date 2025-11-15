"""Simplified forked transformer executor."""

from __future__ import annotations

import os
import time
import traceback
from typing import Dict, Tuple

from .cached_compile import exec_code, check_function_like


def _prepare_deep_structures(
    namespace: Dict[str, object],
    deep_structures_to_unpack: Dict[str, Tuple[object, str]],
) -> None:
    pins = namespace.setdefault("PINS", {})
    for pinname, (value, _celltype) in deep_structures_to_unpack.items():
        namespace[pinname] = value
        pins[pinname] = value


def run_forked(
    name,
    code,
    with_ipython_kernel,
    injector,
    module_workspace,
    identifier,
    namespace,
    deep_structures_to_unpack,
    inputs,
    output_name,
    output_celltype,
    scratch,
    result_queue,
    debug=None,
    tf_checksum=None,
):
    del name, with_ipython_kernel, scratch, debug, tf_checksum
    os.environ["SEAMLESS_FORKED_PROCESS"] = "1"
    start_time = time.time()
    try:
        namespace.pop(output_name, None)
        if deep_structures_to_unpack:
            _prepare_deep_structures(namespace, deep_structures_to_unpack)
        with injector.active_workspace(module_workspace, namespace):
            exec_code(
                code,
                identifier,
                namespace,
                inputs,
                output_name,
                with_ipython_kernel=False,
            )
        try:
            result = namespace[output_name]
        except KeyError:
            function_like = check_function_like(code, identifier)
            if function_like:
                func_name, stmt_count = function_like
                input_params = ",".join(f"{inp}={inp}" for inp in sorted(inputs))
                msg = (
                    f"Transformer defines a single function '{func_name}' and {stmt_count} other statements. "
                    f"Did you forget to assign '{output_name}' (e.g. '{output_name} = {func_name}({input_params})')?"
                )
            else:
                msg = f"Output variable '{output_name}' undefined"
            result_queue.put((1, msg))
            return

        result_queue.put((4, (2, time.time() - start_time)))
        result_queue.put((0, result))
    except Exception:
        result_queue.put((1, traceback.format_exc()))


__all__ = ["run_forked"]
