"""Compile source objects for compiled transformers."""

from __future__ import annotations

from copy import deepcopy
import os
import shlex
import subprocess
import tempfile


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    return list(value)


def complete(module_definition: dict, languages_registry=None) -> dict:
    """Complete a module definition using the compiled language registry."""

    if languages_registry is None:
        import seamless_transformer.languages as languages_registry

    m = deepcopy(module_definition)
    target = m.get("target", "profile")
    m["target"] = target
    m["link_options"] = list(m.get("link_options", []))

    for obj in m["objects"].values():
        lang_def = languages_registry.get_language(obj["language"])
        comp = lang_def.compilation
        obj["compile_mode"] = comp.mode
        obj["compiler_binary"] = comp.compiler
        obj["compile_flag"] = comp.compile_flag
        obj["output_flag"] = comp.output_flag
        obj["language_flag"] = comp.language_flag
        if target in ("release", "profile"):
            options = obj.get("options")
            if options is None:
                options = list(comp.options)
            else:
                options = list(options)
            if target == "profile":
                options += list(comp.profile_options)
            else:
                options += list(comp.release_options)
        else:
            options = obj.get("debug_options")
            if options is None:
                options = list(comp.debug_options)
            else:
                options = list(options)
        obj["options"] = options

    return m


def _run_compiler(obj: dict, source_path: str, output_path: str, include_dir: str):
    command = [obj["compiler_binary"]]
    command += _as_list(obj.get("language_flag"))
    command += _as_list(obj.get("compile_flag"))
    command += _as_list(obj.get("options"))
    if obj.get("language") != "rust":
        command += ["-I", include_dir]
    output_flag = obj.get("output_flag", "-o")
    if output_flag:
        command += _as_list(output_flag)
    command += [output_path, source_path]
    try:
        subprocess.run(
            command,
            cwd=include_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Compilation failed:\n"
            + " ".join(shlex.quote(part) for part in command)
            + "\n"
            + exc.stdout
            + exc.stderr
        ) from exc


def compile(module_definition: dict) -> dict[str, bytes]:
    """Compile all source objects in a completed module definition."""

    with tempfile.TemporaryDirectory() as build_dir:
        public_header = module_definition.get("public_header")
        if public_header is not None:
            with open(os.path.join(build_dir, "public.h"), "w") as f:
                f.write(public_header["code"])

        binaries: dict[str, bytes] = {}
        for objname, obj in module_definition["objects"].items():
            source_path = os.path.join(build_dir, f"{objname}.code")
            suffix = ".a" if obj["compile_mode"] == "archive" else ".o"
            output_path = os.path.join(build_dir, objname + suffix)
            with open(source_path, "w") as f:
                f.write(obj["code"])
            _run_compiler(obj, source_path, output_path, build_dir)
            with open(output_path, "rb") as f:
                binaries[objname + suffix] = f.read()
        return binaries


def _merge_objects(objects: dict[str, bytes], mode: str) -> bytes | dict[str, bytes]:
    """Merge object outputs for legacy-compatible tests."""

    if mode == "object":
        return objects
    if mode == "archive":
        if len(objects) != 1:
            raise ValueError("archive mode requires exactly one archive")
        return next(iter(objects.values()))
    raise NotImplementedError(mode)


__all__ = ["complete", "compile", "_merge_objects"]
