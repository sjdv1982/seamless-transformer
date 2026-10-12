"""Registry of compiled language definitions for seamless-transformer.

Built-in languages (``c``, ``cpp``, ``fortran``, ``rust``) are registered
automatically on import. Additional languages can be registered at runtime
with :func:`define_compiled_language`.

Example — register a custom language::

    from seamless_transformer.languages import define_compiled_language

    define_compiled_language(
        name="mylang",
        compilation={
            "compiler": "mycc",
            "mode": "object",
            "options": ["-O2", "-fPIC"],
            "debug_options": ["-g", "-fPIC"],
            "profile_options": [],
            "release_options": [],
            "compile_flag": "-c",
            "output_flag": "-o",
            "language_flag": "",
        },
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

    Raises ``ValueError`` if ``mode`` is not ``"object"`` or ``"archive"``.
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
    """Register a Python function that executes one transformation language."""

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

    _handler_registry[name] = LanguageHandlerDefinition(
        name=name,
        handler=handler,
        source=source,
        parameters=parameters,
        metavars=defaults,
    )


def get_language_handler(name: str) -> LanguageHandlerDefinition:
    """Return the registered handler definition for the given language."""

    try:
        return _handler_registry[name]
    except KeyError:
        raise KeyError(f"Unknown handler language: {name!r}") from None


def is_handler_language(name: str) -> bool:
    """Return whether name has a registered language handler."""

    return name in _handler_registry


def available_languages() -> dict[str, str]:
    """Return each constructible language and its transformer kind."""

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
    "register_language_handler",
    "run_in_interpreter_thread",
    "_registry",
    "_handler_registry",
]
