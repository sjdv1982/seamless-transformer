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


@dataclass
class LanguageDefinition:
    name: str
    compilation: CompilationConfig


_registry: dict[str, LanguageDefinition] = {}


def define_compiled_language(name: str, compilation: dict) -> None:
    """Register a compiled language definition.

    ``compilation`` is a dict with the following keys:

    - ``compiler`` (str): the compiler binary name (e.g. ``"gcc"``).
    - ``mode`` (str): ``"object"`` (produces ``.o``) or ``"archive"`` (produces ``.a``).
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

    config = CompilationConfig(**compilation)
    if config.mode not in ("object", "archive"):
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


__all__ = [
    "CompilationConfig",
    "LanguageDefinition",
    "define_compiled_language",
    "get_language",
    "is_compiled_language",
    "_registry",
]
