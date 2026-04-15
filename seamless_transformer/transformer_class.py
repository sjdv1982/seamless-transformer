"""Wrap a function in a Seamless transformer."""

from __future__ import annotations

from types import FunctionType, ModuleType
import inspect
from copy import deepcopy
from functools import update_wrapper
from typing import Callable, Generic, Optional, ParamSpec, TypeVar, cast, overload

from seamless import Buffer, ensure_open

from .environment import Environment
from .pretransformation import direct_transformer_to_pretransformation
from .transformation_class import Transformation, transformation_from_pretransformation


P = ParamSpec("P")
R = TypeVar("R")


@overload
def direct(
    func: "Transformer[P, R]", language: None = None
) -> "DirectTransformer[P, R]": ...


@overload
def direct(
    func: Callable[P, R] | str, language: Optional[str] = None
) -> "DirectTransformer[P, R]": ...


def direct(
    func: Callable[P, R] | "Transformer[P, R]" | str, language: Optional[str] = None
) -> "DirectTransformer[P, R]":
    """Execute immediately, returning the result value."""

    if isinstance(func, Transformer):
        result = DirectTransformer.__new__(DirectTransformer)
        for k, v in func.__dict__.items():
            setattr(result, k, deepcopy(v))
        if language is not None:
            result.language = language
    else:
        if language is None:
            language = "python"
        if callable(func):
            if not isinstance(func, FunctionType):
                raise TypeError("func must be a function")
            assert language == "python", language
        result = DirectTransformer(
            func, scratch=False, direct_print=False, local=False, language=language
        )
        if callable(func):
            update_wrapper(result, func)
    return result


def delayed(
    func: Callable[P, R] | str, language: Optional[str] = None
) -> "Transformer[P, R]":
    """Return a Transformation object that can be executed later."""

    if isinstance(func, Transformer):
        result = Transformer.__new__(Transformer)
        for k, v in func.__dict__.items():
            setattr(result, k, v)
        if language is not None:
            result.language = language
    else:
        if language is None:
            language = "python"
        if callable(func):
            if not isinstance(func, FunctionType):
                raise TypeError("func must be a function")
            assert language == "python", language
        result = Transformer(
            func, scratch=False, direct_print=False, local=False, language=language
        )
        if callable(func):
            update_wrapper(result, func)
    return result


class TransformerCore(Generic[P, R]):
    """Shared transformer state and call assembly."""

    def _init_core(
        self,
        *,
        language: str,
        scratch: bool,
        direct_print: bool,
        local: bool,
    ) -> None:
        self._language = language
        self._args = {}
        self._modules = {}
        self._globals = {}
        self._celltypes = {}
        self._environment = Environment()
        self._meta = {"transformer_path": ["tf", "tf"], "local": local}
        self.scratch = scratch
        self.direct_print = direct_print

    def _get_signature(self):
        return None

    def _get_codebuf(self):
        raise NotImplementedError

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, lang):
        if lang is None:
            lang = "python"
        self._language = lang

    @property
    def celltypes(self):
        """The celltypes."""

        return CelltypesWrapper(
            self._celltypes, self._args, fixed=self._get_signature() is not None
        )

    @property
    def args(self):
        """Pre-bound transformer arguments."""

        return ArgsWrapper(
            self._args, self._celltypes, fixed=self._get_signature() is not None
        )

    @property
    def modules(self):
        """Imported Python modules."""

        return ModulesWrapper(self._modules)

    @property
    def globals(self):
        """Global symbols injected via modules.main."""

        return GlobalsWrapper(self._globals)

    @property
    def environment(self) -> Environment:
        """Execution environment for this transformer."""

        return self._environment

    def _bind_arguments(self, *args, **kwargs):
        all_args = self._args.copy()
        signature = self._get_signature()
        if signature is not None:
            all_args.update(signature.bind_partial(*args, **kwargs).arguments)
            return signature.bind(**all_args).arguments
        if len(args) > 0:
            raise TypeError("No function signature: positional arguments not supported")
        all_args.update(kwargs)
        for argname in self._celltypes:
            if argname == "result":
                continue
            if argname not in all_args:
                raise TypeError(f"Missing argument: '{argname}'")
        return all_args

    def __call__(self, *args, **kwargs) -> Transformation[R]:
        """Build a delayed Transformation from the current transformer state."""

        ensure_open("transformer call")
        arguments = self._bind_arguments(*args, **kwargs)
        deps = {
            argname: arg
            for argname, arg in arguments.items()
            if isinstance(arg, Transformation)
        }
        env = self._environment._to_lowlevel()

        meta = deepcopy(self._meta)
        modules = {}
        from .module_builder import (
            build_globals_module_definition,
            get_module_definition,
            merge_module_definitions,
        )

        for module_name, module in self._modules.items():
            if isinstance(module, dict):
                module_definition = module
            else:
                module_definition = get_module_definition(module)
            modules[module_name] = module_definition

        if self._globals:
            globals_def = build_globals_module_definition(self._globals)
            if "main" in modules:
                modules["main"] = merge_module_definitions(modules["main"], globals_def)
            else:
                modules["main"] = globals_def

        pre_transformation = direct_transformer_to_pretransformation(
            self._get_codebuf(),
            meta,
            self._celltypes,
            modules,
            arguments,
            env,
            language=self.language,
        )
        return cast(
            Transformation[R],
            transformation_from_pretransformation(
                pre_transformation,
                upstream_dependencies=deps,
                meta=meta,
                scratch=self.scratch,
                tf_dunder={},
            ),
        )

    @property
    def meta(self):
        """Transformation metadata."""

        return self._meta

    @meta.setter
    def meta(self, meta: dict):
        self._meta.update(meta)
        for k in list(self._meta.keys()):
            if self._meta[k] is None and k != "local":
                self._meta.pop(k)

    @property
    def scratch(self) -> bool:
        """If True, the transformation result buffer will not be saved."""

        return self._scratch

    @scratch.setter
    def scratch(self, value: bool):
        self._scratch = value

    @property
    def allow_input_fingertip(self) -> bool:
        """If True, inputs may be fingertipped when resolving their buffers."""

        return bool(self._meta.get("allow_input_fingertip", False))

    @allow_input_fingertip.setter
    def allow_input_fingertip(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(type(value))
        if value:
            self.meta = {"allow_input_fingertip": True}
        else:
            self._meta.pop("allow_input_fingertip", None)

    @property
    def direct_print(self):
        """Print stdout/stderr directly instead of only storing logs."""

        return self._meta.get("__direct_print__", False)

    @direct_print.setter
    def direct_print(self, value):
        if not isinstance(value, bool) and value is not None:
            raise TypeError(type(value))
        self.meta = {"__direct_print__": value}

    @property
    def driver(self) -> bool:
        """Marks the transformer as a driver script."""

        return self._meta.get("driver", False)

    @driver.setter
    def driver(self, value):
        if not isinstance(value, bool) and value is not None:
            raise TypeError(type(value))
        self.meta = {"driver": value}

    @property
    def local(self) -> bool | None:
        """Local execution preference."""

        return self.meta.get("local")

    @local.setter
    def local(self, value: bool | None):
        self.meta["local"] = value


class PythonMixin(Generic[P, R]):
    """Python and text-source behavior for ordinary transformers."""

    def __init__(
        self,
        code: Callable[P, R] | str,
        *,
        language: str,
        scratch: bool,
        direct_print: bool,
        local: bool,
    ):
        self._init_core(
            language=language,
            scratch=scratch,
            direct_print=direct_print,
            local=local,
        )
        self._set_code(code)
        if callable(code):
            update_wrapper(self, code)

    def _set_code(self, code: Callable[P, R] | str):
        from .getsource import getsource

        signature = None
        if callable(code):
            assert isinstance(code, FunctionType)
            signature = inspect.signature(code)
            code = getsource(code)
            codebuf = Buffer(code, "python")
            self._codebuf = codebuf
            self._celltypes = {k: "mixed" for k in signature.parameters}
            self._celltypes["result"] = "mixed"
        else:
            assert isinstance(code, str)
            self._codebuf = Buffer(code, "text")
        self._signature = signature

    def _get_signature(self):
        return self._signature

    def _get_codebuf(self):
        return self._codebuf

    @property
    def code(self):
        return self._codebuf

    @code.setter
    def code(self, code: Callable[P, R] | str):
        return self._set_code(code)

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, lang):
        if lang is None:
            lang = "python"
        self._language = lang
        if lang != "python":
            self._signature = None


class Transformer(PythonMixin[P, R], TransformerCore[P, R]):
    """Ordinary Python/bash transformer."""


class DirectTransformer(Transformer[P, R]):
    """Transformer that computes immediately."""

    def __call__(self, *args, **kwargs) -> R:
        tf = super().__call__(*args, **kwargs)
        tf._compute(api_origin="call")
        return tf.run()


class CelltypesWrapper:
    """Wrapper around an imperative transformer's celltypes."""

    def __init__(self, celltypes, args, fixed):
        self._celltypes = celltypes
        self._args = args
        self._fixed = fixed

    def __getattr__(self, attr):
        return self._celltypes[attr]

    def __getitem__(self, key):
        return self._celltypes[key]

    def __setattr__(self, attr, value):
        if attr.startswith("_"):
            return super().__setattr__(attr, value)
        return self.__setitem__(attr, value)

    def __setitem__(self, key, value):
        from seamless.checksum.celltypes import celltypes

        if key not in self._celltypes:
            if self._fixed:
                raise AttributeError(key)
            self._celltypes[key] = value
            if "result" not in self._celltypes:
                self._celltypes["result"] = "mixed"

        if isinstance(value, type):
            value = value.__name__
        value = str(value)
        if key == "result":
            if value in ("deepfolder", "module"):
                raise TypeError(f"result celltype cannot be '{value}'")
            all_celltypes = celltypes + ["deepcell", "folder"]
        else:
            all_celltypes = celltypes + ["deepcell", "deepfolder", "folder", "module"]
        if value not in all_celltypes:
            raise TypeError(value, all_celltypes)
        old_arg = self._args.get(key)
        if old_arg is not None:
            pass
        self._celltypes[key] = value

    def __delattr__(self, attr: str) -> None:
        if attr.startswith("_"):
            return super().__delattr__(attr)
        return self.__delitem__(attr)

    def __delitem__(self, key) -> None:
        if self._fixed or key not in self._celltypes:
            raise AttributeError(key)
        del self._celltypes[key]
        if key in self._args:
            del self._args[key]

    def __dir__(self):
        return sorted(self._celltypes.keys())

    def __str__(self):
        return str(self._celltypes)

    def __repr__(self):
        return str(self)


class ArgsWrapper:
    """Wrapper around an imperative transformer's arguments."""

    def __init__(self, args, celltypes, fixed):
        self._args = args
        self._celltypes = celltypes
        self._fixed = fixed

    def __getattr__(self, attr):
        return self._args.get(attr)

    def __getitem__(self, key):
        return self._args.get(key)

    def __setattr__(self, attr, value):
        if attr.startswith("_"):
            return super().__setattr__(attr, value)
        return self.__setitem__(attr, value)

    def __setitem__(self, key, value):
        if key == "result":
            raise AttributeError(key)
        if key not in self._celltypes:
            if self._fixed:
                raise AttributeError(key)
            self._celltypes[key] = "mixed"
            if "result" not in self._celltypes:
                self._celltypes["result"] = "mixed"
        self._args[key] = value

    def __delattr__(self, attr: str) -> None:
        if attr.startswith("_"):
            return super().__delattr__(attr)
        return self.__delitem__(attr)

    def __delitem__(self, key) -> None:
        if self._fixed or key not in self._celltypes:
            raise AttributeError(key)
        del self._celltypes[key]
        if key in self._args:
            del self._args[key]

    def __dir__(self):
        return sorted(self._args.keys())

    def __str__(self):
        return str(self._args)

    def __repr__(self):
        return str(self)


class ModulesWrapper:
    """Wrapper around an imperative transformer's imported modules."""

    def __init__(self, modules):
        self._modules = modules

    def __getattr__(self, attr):
        return self._modules[attr]

    def __setattr__(self, attr, value):
        if attr.startswith("_"):
            return super().__setattr__(attr, value)
        if not isinstance(value, (ModuleType, dict)):
            raise TypeError(type(value))
        self._modules[attr] = value

    def __dir__(self):
        return sorted(self._modules.keys())

    def __str__(self):
        return str(self._modules)

    def __repr__(self):
        return str(self)


class GlobalsWrapper:
    """Wrapper around an imperative transformer's global namespace."""

    def __init__(self, globals_dict):
        self._globals = globals_dict

    def __getattr__(self, attr):
        return self._globals.get(attr)

    def __getitem__(self, key):
        return self._globals.get(key)

    def __setattr__(self, attr, value):
        if attr.startswith("_"):
            return super().__setattr__(attr, value)
        return self.__setitem__(attr, value)

    def __setitem__(self, key, value):
        self._globals[key] = value

    def __delattr__(self, attr: str) -> None:
        if attr.startswith("_"):
            return super().__delattr__(attr)
        return self.__delitem__(attr)

    def __delitem__(self, key) -> None:
        self._globals.pop(key, None)

    def __dir__(self):
        return sorted(self._globals.keys())

    def __str__(self):
        return str(self._globals)

    def __repr__(self):
        return str(self)


__all__ = [
    "direct",
    "delayed",
    "TransformerCore",
    "PythonMixin",
    "Transformer",
    "DirectTransformer",
    "CelltypesWrapper",
    "ArgsWrapper",
    "ModulesWrapper",
    "GlobalsWrapper",
]
