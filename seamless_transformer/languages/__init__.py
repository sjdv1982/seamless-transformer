"""Registries and helpers for languages accepted by Seamless transformers.

``python`` and ``bash`` are the built-in native languages. Register compiled
languages with ``define_compiled_language`` and languages executed by Python
handlers with ``register_language_handler``. ``available_languages()`` returns
each language accepted by the ``Transformer`` factory and its kind
(``"native"``, ``"compiled"`` or ``"handler"``). A language name belongs to
one kind.

Built-in compiled languages and the ``r`` and ``julia`` handlers register
when this module is imported. Registration does not import or start R or Julia,
or load ``ryp`` or ``juliacall``.

``run_in_interpreter_thread()`` runs a callable on one process-wide thread
associated with its name, for embedded interpreters that need a stable calling
thread. Calls from that thread run inline. Set
``SEAMLESS_INTERPRETER_THREAD=0`` to run on the calling thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar


_cli_execution_context = ContextVar("seamless_cli_execution_context", default=(None, None))


@contextmanager
def _use_cli_execution_context(tf_checksum, should_abort):
    token = _cli_execution_context.set((tf_checksum, should_abort))
    try:
        yield
    finally:
        _cli_execution_context.reset(token)


def execute_cli(command, pins_, PINS, FILESYSTEM, OUTPUTPIN, environment):
    """Run a language command through the shared CLI job machinery."""
    from collections.abc import Mapping

    if environment is None:
        environment = {}
    if not isinstance(environment, Mapping):
        raise TypeError("transformation environment must be a mapping")
    from seamless_transformer.execute_bash import execute_bash

    if environment.get("docker") is not None:
        import os

        from seamless import CacheMissError
        from seamless_transformer.execute_bash import _container_can_read
        from seamless_transformer.transformation_namespace import (
            _get_read_folder_directories,
        )

        read_directories = _get_read_folder_directories()
        for pinname, entry in FILESYSTEM.items():
            if not isinstance(entry, Mapping) or entry.get("celltype") != "deepfolder":
                continue
            value = PINS.get(pinname)
            if not isinstance(value, Mapping):
                continue
            for member, path in value.items():
                if not isinstance(path, (str, os.PathLike)):
                    raise CacheMissError(
                        f"deepfolder member {member!r} is not a stored buffer path"
                    )
                if not os.path.isfile(path) or not _container_can_read(
                    os.fspath(path), read_directories
                ):
                    raise CacheMissError(
                        f"deepfolder member {member!r}: buffer file is a link "
                        "which the container cannot follow"
                    )

    tf_checksum, should_abort = _cli_execution_context.get()
    return execute_bash(
        command,
        pins_,
        environment.get("conda_environment", ""),
        PINS,
        FILESYSTEM,
        OUTPUTPIN,
        which_=environment.get("which", ()),
        docker_=environment.get("docker"),
        tf_checksum_=tf_checksum,
        should_abort_=should_abort,
    )


def check_in_memory_environment(environment, language):
    """Check an in-memory backend's conda, Docker, and required-binary rules.

    A conda setting is a requirement on the current process; this function
    never activates an environment. Docker is unsupported by in-memory
    backends. Required binaries are checked on this process's PATH.
    """
    import os
    import shutil
    from collections.abc import Mapping

    if environment is None:
        environment = {}
    if not isinstance(environment, Mapping):
        raise TypeError("transformation environment must be a mapping")

    language = str(language)
    if "docker" in environment:
        hint = '; use BACKEND="cli"' if language in {"python", "r", "julia"} else ""
        raise RuntimeError(
            f"The in-memory backend for {language!r} cannot use a Docker image{hint}"
        )

    conda_environment = environment.get("conda_environment")
    if conda_environment is not None:
        if not isinstance(conda_environment, str):
            raise TypeError("conda_environment must be a string")
        if "/" in conda_environment:
            current = os.environ.get("CONDA_PREFIX")
            matches = current is not None and os.path.realpath(current) == os.path.realpath(
                conda_environment
            )
            current_description = current if current is not None else "<unset>"
        else:
            current = os.environ.get("CONDA_DEFAULT_ENV")
            matches = current == conda_environment
            current_description = current if current is not None else "<unset>"
        if not matches:
            raise RuntimeError(
                f"The in-memory backend for {language!r} requires conda environment "
                f"{conda_environment!r}, but the current process is in "
                f"{current_description!r}"
            )

    required = environment.get("which", ())
    if required is None:
        required = ()
    elif not isinstance(required, (tuple, list)):
        raise TypeError("which must be a list or tuple of binary names")
    missing = []
    for binary in required:
        if not isinstance(binary, str):
            raise TypeError("which binary names must be strings")
        if shutil.which(binary) is None:
            missing.append(binary)
    if missing:
        raise RuntimeError(
            "Required environment binaries are unavailable: " + ", ".join(missing)
        )


def _validate_backend_value(value, language):
    if type(value) is not str or value not in ("cli", "mem"):
        raise ValueError(
            f"Invalid BACKEND for language {language!r}: {value!r}; "
            "expected 'cli' or 'mem'"
        )


@dataclass
class CompilationConfig:
    compiler: str
    mode: str
    options: list[str] = field(default_factory=list)
    debug_options: list[str] = field(default_factory=list)
    profile_options: list[str] = field(default_factory=list)
    release_options: list[str] = field(default_factory=list)
    compile_flag: str = ""
    output_flag: str = "-o"
    language_flag: str = ""
    include_flag: str = "-I"
    extension: str = ""


@dataclass
class LanguageDefinition:
    name: str
    compilation: CompilationConfig


_registry: dict[str, LanguageDefinition] = {}
_handler_registry: dict[str, LanguageHandlerDefinition] = {}
_NATIVE_LANGUAGES = frozenset({"python", "bash"})


def define_compiled_language(name: str, compilation: dict) -> None:
    """Register a compiled language definition.

    ``compilation`` is a dict with the following keys:

    - ``compiler`` (str): the compiler binary name (e.g. ``"gcc"``).
    - ``mode`` (str): ``"object"`` (produces ``.o``), ``"archive"`` (produces ``.a``),
      or ``"package"`` (multi-file package compiled to ``.a``, e.g. Go).
    - ``include_flag`` (str): flag used to pass the build include directory to the compiler
      (e.g. ``"-I"`` for gcc/gfortran); empty string disables include-dir injection.
      Defaults to ``"-I"``.
    - ``extension`` (str): source-file extension required by the compiler (e.g. ``"go"``);
      empty string falls back to ``"code"``.
    - ``options`` (list[str]): default (profile/release) compiler flags.
    - ``debug_options`` (list[str]): flags used for debug builds.
    - ``profile_options`` (list[str]): extra flags added on top of ``options`` for profile target.
    - ``release_options`` (list[str]): flags used for release builds.
    - ``compile_flag`` (str): flag that means "compile only, no link" (e.g. ``"-c"`` for gcc).
    - ``output_flag`` (str): flag for the output file (e.g. ``"-o"``).
    - ``language_flag`` (str): flag to tell the compiler which language to use
      (e.g. ``"-x c"`` for gcc); empty string if not needed.

    Raises ``ValueError`` if ``mode`` is not ``"object"``, ``"archive"`` or
    ``"package"``.
    """

    if name in _NATIVE_LANGUAGES or name in _handler_registry:
        raise ValueError(f"Language {name!r} is already registered by another kind")
    config = CompilationConfig(**compilation)
    if config.mode not in ("object", "archive", "package"):
        raise ValueError(f"Unsupported compile mode: {config.mode!r}")
    _registry[name] = LanguageDefinition(name=name, compilation=config)


def get_language(name: str) -> LanguageDefinition:
    """Look up a registered compiled language by name.

    Raises ``KeyError`` if the language has not been registered.
    """

    if name not in _registry:
        raise KeyError(f"Unknown compiled language: {name!r}")
    return _registry[name]


def is_compiled_language(name: str) -> bool:
    """Return True if ``name`` is a registered compiled language."""
    return name in _registry


from . import native  # noqa: E402,F401



@dataclass(frozen=True)
class LanguageHandlerDefinition:
    """Registered source and defaults for a non-native transformer language."""

    name: str
    handler: object
    source: str
    parameters: tuple[str, ...]
    metavars: dict


_HANDLER_ARGUMENTS = frozenset(
    {
        "code",
        "pins",
        "filesystem",
        "output_name",
        "output_celltype",
        "metavars",
        "environment",
        "process_lock",
    }
)
def register_language_handler(name, handler, *, metavars=None) -> None:
    """Register a Python function as the executor for a transformer language.

    The name must not be native or compiled. ``handler`` must be one named,
    source-retrievable function without a closure. Its supported parameters
    are passed by keyword. ``metavars`` maps setting names to their defaults.
    Registering the same handler language again replaces its registration.
    """

    import ast
    import inspect
    from collections.abc import Mapping

    from ..cached_compile import analyze_code
    from ..getsource import getsource

    if not isinstance(name, str) or not name:
        raise ValueError("language handler name must be a non-empty string")
    if name in _NATIVE_LANGUAGES or name in _registry:
        raise ValueError(f"Language {name!r} is already registered by another kind")
    if not inspect.isfunction(handler):
        raise TypeError("language handler must be a Python function")
    if handler.__name__ == "<lambda>":
        raise ValueError(
            "language handler must be a named Python function; "
            "lambda handlers do not have a stable source definition"
        )
    if handler.__closure__:
        raise ValueError("language handler must not close over external variables")

    try:
        source = getsource(handler)
    except (OSError, TypeError, ValueError, SyntaxError) as exc:
        raise ValueError(
            f"could not retrieve valid source for language handler {name!r}; "
            "use a named function defined in source code"
        ) from exc
    if not isinstance(source, str):
        raise ValueError("could not retrieve source for language handler")
    try:
        mode, function_name = analyze_code(source, f"language-handler-{name}")
        tree = ast.parse(source)
    except (SyntaxError, TypeError) as exc:
        raise ValueError(
            f"could not parse source for language handler {name!r}: {exc}"
        ) from exc
    if mode != "function" or function_name is None:
        raise ValueError(
            "language handler source must be exactly one named Python function"
        )

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError(
            "language handler source must be exactly one named Python function"
        )
    function = tree.body[0]
    if function.args.posonlyargs or function.args.vararg or function.args.kwarg:
        raise ValueError("language handler parameters must be passed by keyword")
    parameter_nodes = function.args.args + function.args.kwonlyargs
    parameters = tuple(parameter.arg for parameter in parameter_nodes)
    unsupported = sorted(set(parameters) - _HANDLER_ARGUMENTS)
    if unsupported:
        raise ValueError(
            "unsupported language handler parameter(s): " + ", ".join(unsupported)
        )

    if metavars is None:
        metavars = {}
    if not isinstance(metavars, Mapping):
        raise TypeError("metavars must be a mapping")
    defaults = dict(metavars)
    if any(not isinstance(key, str) for key in defaults):
        raise TypeError("metavar names must be strings")
    if "BACKEND" in defaults:
        _validate_backend_value(defaults["BACKEND"], name)

    _handler_registry[name] = LanguageHandlerDefinition(
        name=name,
        handler=handler,
        source=source,
        parameters=parameters,
        metavars=defaults,
    )


def get_language_handler(name: str) -> LanguageHandlerDefinition:
    """Return the registered handler definition for ``name``.

    Raises ``KeyError`` when no handler is registered for that name.
    """

    try:
        return _handler_registry[name]
    except KeyError:
        raise KeyError(f"Unknown handler language: {name!r}") from None


def is_handler_language(name: str) -> bool:
    """Return whether name has a registered language handler."""

    return name in _handler_registry


def available_languages() -> dict[str, str]:
    """Return Transformer-constructible languages mapped to their kind.

    Each value is ``"native"``, ``"compiled"`` or ``"handler"``.
    """

    result = {name: "native" for name in _NATIVE_LANGUAGES}
    result.update({name: "compiled" for name in _registry})
    result.update({name: "handler" for name in _handler_registry})
    return result


from . import handlers  # noqa: E402,F401
from .interpreter_thread import run_in_interpreter_thread


__all__ = [
    "CompilationConfig",
    "LanguageDefinition",
    "LanguageHandlerDefinition",
    "define_compiled_language",
    "get_language",
    "get_language_handler",
    "is_compiled_language",
    "is_handler_language",
    "available_languages",
    "check_in_memory_environment",
    "execute_cli",
    "register_language_handler",
    "run_in_interpreter_thread",
    "_registry",
    "_handler_registry",
]
