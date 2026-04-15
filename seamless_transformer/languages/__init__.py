"""Compiled language registry."""

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
    """Register a compiled language definition."""

    config = CompilationConfig(**compilation)
    if config.mode not in ("object", "archive"):
        raise ValueError(f"Unsupported compile mode: {config.mode!r}")
    _registry[name] = LanguageDefinition(name=name, compilation=config)


def get_language(name: str) -> LanguageDefinition:
    """Look up a registered compiled language."""

    if name not in _registry:
        raise KeyError(f"Unknown compiled language: {name!r}")
    return _registry[name]


def is_compiled_language(name: str) -> bool:
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
