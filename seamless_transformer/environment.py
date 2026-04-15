"""Execution environment for transformer calls."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class Environment:
    """Execution environment for an individual transformer."""

    _props = ("_conda", "_conda_env_name", "_which", "_powers", "_docker")

    def __init__(self):
        self._conda = None
        self._conda_env_name = None
        self._which = None
        self._powers = None
        self._docker = None

    def _save(self) -> dict[str, Any] | None:
        state = {}
        for prop in self._props:
            value = getattr(self, prop)
            if value is not None:
                state[prop[1:]] = deepcopy(value)
        return state or None

    def _load(self, state: dict[str, Any] | None):
        if state is None:
            state = {}
        if not isinstance(state, dict):
            raise TypeError(type(state))
        for prop in self._props:
            value = state.get(prop[1:])
            if value is not None:
                setattr(self, prop, deepcopy(value))

    def _to_lowlevel(self) -> dict[str, Any] | None:
        result = {}
        if self._which is not None:
            result["which"] = deepcopy(self._which)
        if self._conda is not None:
            result["conda"] = yaml.safe_load(self._conda)
        if self._conda_env_name is not None:
            result["conda_environment"] = self._conda_env_name
        if self._powers is not None:
            result["powers"] = deepcopy(self._powers)
        if self._docker is not None:
            result["docker"] = deepcopy(self._docker)
        return result or None

    def set_conda(self, conda):
        """Set a conda environment spec from YAML text, a path, or a file object."""

        if hasattr(conda, "read"):
            conda_text = conda.read()
        elif isinstance(conda, Path):
            conda_text = conda.read_text()
        elif isinstance(conda, str):
            expanded = Path(conda).expanduser()
            if "\n" not in conda and expanded.exists():
                conda_text = expanded.read_text()
            else:
                conda_text = conda
        else:
            raise TypeError(type(conda))
        if not isinstance(conda_text, str):
            raise TypeError(type(conda_text))
        parsed = yaml.safe_load(conda_text)
        if not isinstance(parsed, dict):
            raise TypeError("conda specification must be a YAML mapping")
        if "dependencies" not in parsed:
            raise ValueError("conda specification must contain 'dependencies'")
        self._conda = conda_text

    def get_conda(self):
        return deepcopy(self._conda)

    def set_conda_env(self, conda_env_name):
        if conda_env_name is not None and not isinstance(conda_env_name, str):
            raise TypeError(type(conda_env_name))
        self._conda_env_name = conda_env_name

    def get_conda_env(self):
        return self._conda_env_name

    def set_which(self, which):
        if which is None:
            self._which = None
            return
        if not isinstance(which, (list, tuple)):
            raise TypeError(type(which))
        which2 = []
        for item in which:
            if not isinstance(item, str):
                raise TypeError(type(item))
            which2.append(item)
        self._which = which2

    def get_which(self):
        return deepcopy(self._which)

    def set_powers(self, powers):
        if powers is None:
            self._powers = None
            return
        if not isinstance(powers, dict):
            raise TypeError(type(powers))
        self._powers = deepcopy(powers)

    def get_powers(self):
        return deepcopy(self._powers)

    def set_docker(self, docker: dict):
        if docker is None:
            self._docker = None
            return
        if not isinstance(docker, dict):
            raise TypeError(type(docker))
        if "name" not in docker:
            raise ValueError("docker specification must contain 'name'")
        self._docker = deepcopy(docker)

    def get_docker(self):
        return deepcopy(self._docker)


__all__ = ["Environment"]
