"""Wrap a function in a Seamless transformer."""

from __future__ import annotations

from types import FunctionType
import inspect
from copy import deepcopy
from functools import update_wrapper
from typing import Callable, Generic, Optional, ParamSpec, TypeVar, cast, overload

from seamless import Buffer, ensure_open
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
        assert language is None
        result = DirectTransformer.__new__(DirectTransformer)
        for k, v in func.__dict__.items():
            setattr(result, k, deepcopy(v))
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
        assert language is None
        result = Transformer.__new__(Transformer)
        for k, v in func.__dict__.items():
            setattr(result, k, v)
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


class Transformer(Generic[P, R]):
    """Transformer.
    Transformers can be called as normal functions, but
    the source code of the function and the arguments are converted
    into a Seamless Transformation that is returned."""

    def __init__(
        self,
        code: Callable[P, R] | str,
        *,
        language: str,
        scratch: bool,
        direct_print: bool,
        local: bool,
    ):
        """Transformer.
        Transformers can be called as normal functions, but
        the source code of the function and the arguments are converted
        into a Seamless Transformation that is returned.types

            Parameters:

            - local. If True, transformations are executed in the local
                        Seamless instance.
                    If False (default), they are executed remotely if possible.

            - scratch.
                    If True, only the checksum is preserved. This is for cases where
                      the result is bulky, but can be recomputed easily
                    If False (default), the buffers are preserved, so that the value
                      can be accessed if needed.


            - direct_print: If True, it is attempted to print stdout and stderr
                    while the transformation runs.

            Attributes:

            - meta. Accesses all meta-information (including local)

            - driver. If True, marks the transformer as a driver script so
                    its transformations bypass Dask throttling.

            - celltypes. Returns a wrapper where you can set the celltypes
                    of the individual transformer args.
                The syntax is: Transformer.celltypes.a = "text"
                (or Transformer.celltypes["a"] = "text")
                for arg "a".

            - modules: Returns a wrapper where you can define Python modules
                to be imported into the transformation

            - environment  ...

        """
        self._language = language
        self._args = {}
        self._modules = {}
        self._globals = {}
        self._celltypes = {}
        self._set_code(code)
        """
        STUB
        self._environment = Environment(self)
        self._environment_state = None
        """
        self._meta = {"transformer_path": ["tf", "tf"], "local": local}
        self.scratch = scratch
        self.direct_print = direct_print
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

    @property
    def celltypes(self):
        """The celltypes"""
        return CelltypesWrapper(
            self._celltypes, self._args, fixed=self._signature is not None
        )

    @property
    def args(self):
        """The arguments"""
        return ArgsWrapper(
            self._args, self._celltypes, fixed=self._signature is not None
        )

    @property
    def modules(self):
        """The imported modules"""
        return ModulesWrapper(self._modules)

    @property
    def globals(self):
        """Global symbols injected via modules.main."""
        return GlobalsWrapper(self._globals)

    '''
    STUB
    @property
    def environment(self) -> "Environment":
        """Computing environment to execute transformations in"""
        return self._environment
    /STUB
    '''

    def __call__(self, *args, **kwargs) -> Transformation[R]:
        """
        from seamless.workflow.core.direct.module import get_module_definition
        """

        ensure_open("transformer call")
        all_args = self._args.copy()
        if self._signature is not None:
            all_args.update(self._signature.bind_partial(*args, **kwargs).arguments)
            arguments = self._signature.bind(**all_args).arguments
        else:
            if len(args) > 0:
                raise TypeError(
                    "No function signature: positional arguments not supported"
                )
            all_args.update(kwargs)
            for argname in self._celltypes:
                if argname == "result":
                    continue
                if argname not in all_args:
                    raise TypeError(f"Missing argument: '{argname}'")
            arguments = all_args
        deps = {
            argname: arg
            for argname, arg in arguments.items()
            if isinstance(arg, Transformation)
        }
        env = None  # environment handling not ported

        meta = deepcopy(self._meta)
        modules = {}
        from .module_builder import (
            build_globals_module_definition,
            get_module_definition,
            merge_module_definitions,
        )
        """
        STUB
        for module_name, module in self._modules.items():
            if isinstance(module, dict):
                module_definition = module
            else:
                module_definition = get_module_definition(module)
            modules[module_name] = module_definition
        /STUB
        """
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
            self._codebuf,
            meta,
            self._celltypes,
            modules,
            arguments,
            env,
            language=self.language,
        )
        tf_dunder = {}
        return cast(
            Transformation[R],
            transformation_from_pretransformation(
                pre_transformation,
                upstream_dependencies=deps,
                meta=meta,
                scratch=self.scratch,
                tf_dunder=tf_dunder,
            ),
        )

    @property
    def meta(self):
        """Transformation metadata"""
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
            if "allow_input_fingertip" in self._meta:
                self._meta.pop("allow_input_fingertip", None)

    @property
    def direct_print(self):
        """Causes the transformer to directly print any messages,
        instead of buffering them and storing them in Transformer.logs.
        If this value is None, direct print is True if debugging is enabled."""
        return self._meta.get("__direct_print__", False)

    @direct_print.setter
    def direct_print(self, value):
        if not isinstance(value, bool) and value is not None:
            raise TypeError(type(value))
        self.meta = {"__direct_print__": value}

    @property
    def driver(self) -> bool:
        """Marks the transformer as a driver script (bypasses Dask throttling)."""
        return self._meta.get("driver", False)

    @driver.setter
    def driver(self, value):
        if not isinstance(value, bool) and value is not None:
            raise TypeError(type(value))
        self.meta = {"driver": value}

    @property
    def local(self) -> bool | None:
        """Local execution.
        If True, transformations are executed in the local Seamless instance.
        If False, they are delegated to an assistant.
        If None (default),
        an assistant is tried first and local execution is a fallback."""
        return self.meta.get("local")

    @local.setter
    def local(self, value: bool | None):
        self.meta["local"] = value


class DirectTransformer(Transformer[P, R]):
    """Transformer that can be called and gives an immediate result"""

    def __call__(self, *args, **kwargs) -> R:
        """
        from seamless.workflow.core.direct.module import get_module_definition
        """
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
            pass  # TODO: verify compatibility with new celltype, but only inside a workflow
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
    """Wrapper around an imperative transformer's celltypes."""

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
        from seamless.checksum.celltypes import celltypes

        if key == "result":
            raise AttributeError(key)
        if key not in self._celltypes:
            if self._fixed:
                raise AttributeError(key)
            self._celltypes[key] = "mixed"
            if "result" not in self._celltypes:
                self._celltypes["result"] = "mixed"

        celltype = self._celltypes[key]
        # TODO: verify compatibility with celltype, but only inside a workflow
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
        from types import ModuleType

        """
        STUB
        from seamless.workflow.highlevel.Module import Module
        /STUB
        """

        if attr.startswith("_"):
            return super().__setattr__(attr, value)
        """
        STUB
        if isinstance(value, Module):
            module_definition = value.module_definition
            if module_definition is None:
                raise RuntimeError("Seamless Module has not been translated yet")
            self._modules[attr] = module_definition
        /STUB
        """
        if 0:
            pass
        else:
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
