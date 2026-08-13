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
from .builder_snapshot import TransformerBuilderSnapshot


P = ParamSpec("P")
R = TypeVar("R")


def _snapshot_modules(modules):
    """Copy module mappings without attempting to pickle module objects."""

    return {
        name: value if isinstance(value, ModuleType) else deepcopy(value)
        for name, value in modules.items()
    }


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
        self._optional_pins = set()
        self._environment = Environment()
        self._meta = {"transformer_path": ["tf", "tf"], "local": local}
        self._workflow_backend = None
        self._workflow_callable = None
        self._refholds_released = False
        self.scratch = scratch
        self.direct_print = direct_print
        from seamless.reference_lifecycle import register_refholder

        register_refholder(self)

    def _get_signature(self):
        return None

    def _get_codebuf(self):
        raise NotImplementedError

    def _snapshot_for_call(self) -> TransformerBuilderSnapshot:
        if self._workflow_backend is not None:
            return self._workflow_backend.snapshot_for_call()
        return TransformerBuilderSnapshot(
            codebuf=self._get_codebuf(),
            language=self.language,
            celltypes=deepcopy(self._celltypes),
            optional_pins=frozenset(self._optional_pins),
            args=deepcopy(self._args),
            modules=_snapshot_modules(self._modules),
            globals=deepcopy(self._globals),
            meta=deepcopy(self._meta),
            environment=self._environment._to_lowlevel(),
            scratch=bool(self.scratch),
            direct_print=bool(self.direct_print),
            local=self.local,
            call_mode="direct" if isinstance(self, DirectTransformer) else "delayed",
            callable=self._workflow_callable,
            signature=self._get_signature(),
        )

    @property
    def language(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.language
        return self._language

    @language.setter
    def language(self, lang):
        if getattr(self, "_workflow_backend", None) is not None:
            self._workflow_backend.language = lang
            return
        if lang is None:
            lang = "python"
        self._language = lang

    @property
    def celltypes(self):
        """The celltypes."""

        if self._workflow_backend is not None:
            return self._workflow_backend.celltypes
        return CelltypesWrapper(
            self._celltypes, self._args, fixed=self._get_signature() is not None
        )

    @property
    def args(self):
        """Pre-bound transformer arguments."""

        if self._workflow_backend is not None:
            return self._workflow_backend.args
        return ArgsWrapper(
            self, self._args, self._celltypes, fixed=self._get_signature() is not None
        )

    @property
    def pins(self):
        """Pre-bound transformer inputs."""

        if self._workflow_backend is not None:
            return self._workflow_backend.pins
        return self.args

    @property
    def modules(self):
        """Imported Python modules."""

        if self._workflow_backend is not None:
            return self._workflow_backend.modules
        return ModulesWrapper(self, self._modules)

    @property
    def globals(self):
        """Global symbols injected via modules.main."""

        if self._workflow_backend is not None:
            return self._workflow_backend.globals
        return GlobalsWrapper(self._globals)

    @property
    def optional_pins(self) -> set[str]:
        """Input pins where JSON null means absence.

        Connected optional pins still compute and still fail on upstream errors.
        Optional pins can be tricky: for these pins, JSON null is reserved as
        absence and only plain/mixed pins can use that absence value.
        """

        if self._workflow_backend is not None:
            return self._workflow_backend.optional_pins
        return self._optional_pins

    @optional_pins.setter
    def optional_pins(self, value) -> None:
        if getattr(self, "_workflow_backend", None) is not None:
            self._workflow_backend.optional_pins = value
            return
        self._optional_pins = set(value or ())

    @property
    def environment(self) -> Environment:
        """Execution environment for this transformer."""

        if self._workflow_backend is not None:
            return self._workflow_backend.environment
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

    @staticmethod
    def _bind_snapshot_arguments(snapshot, args, kwargs):
        all_args = deepcopy(snapshot.args)
        signature = snapshot.signature
        if signature is not None:
            all_args.update(signature.bind_partial(*args, **kwargs).arguments)
            return signature.bind(**all_args).arguments
        if args:
            raise TypeError("No function signature: positional arguments not supported")
        all_args.update(kwargs)
        for argname in snapshot.celltypes:
            if argname == "result":
                continue
            if argname not in all_args and argname not in snapshot.optional_pins:
                raise TypeError(f"Missing argument: '{argname}'")
        return all_args

    def _build_from_snapshot(self, snapshot, *args, **kwargs) -> Transformation[R]:
        ensure_open("transformer call")
        arguments = self._bind_snapshot_arguments(snapshot, args, kwargs)
        from seamless import Expression

        deps = {
            argname: arg
            for argname, arg in arguments.items()
            if isinstance(arg, (Transformation, Expression))
        }
        from .module_builder import (
            build_globals_module_definition,
            get_module_definition,
            merge_module_definitions,
        )

        modules = {}
        for module_name, module in snapshot.modules.items():
            if isinstance(module, dict):
                module_definition = deepcopy(module)
            else:
                module_definition = get_module_definition(module)
            modules[module_name] = module_definition
        if snapshot.globals:
            globals_def = build_globals_module_definition(snapshot.globals)
            if "main" in modules:
                modules["main"] = merge_module_definitions(modules["main"], globals_def)
            else:
                modules["main"] = globals_def

        pre_transformation = direct_transformer_to_pretransformation(
            snapshot.codebuf,
            deepcopy(snapshot.meta),
            deepcopy(snapshot.celltypes),
            modules,
            arguments,
            deepcopy(snapshot.environment),
            language=snapshot.language,
            optional_pins=snapshot.optional_pins,
        )
        return cast(
            Transformation[R],
            transformation_from_pretransformation(
                pre_transformation,
                upstream_dependencies=deps,
                meta=deepcopy(snapshot.meta),
                scratch=snapshot.scratch,
                tf_dunder={},
            ),
        )

    def __call__(self, *args, **kwargs) -> Transformation[R]:
        """Build a delayed Transformation from the current transformer state."""
        return self._build_from_snapshot(self._snapshot_for_call(), *args, **kwargs)

    def transformation(self):
        return self()

    get_transformation = transformation

    @property
    def result(self):
        if self._workflow_backend is None:
            raise AttributeError("result is only available for bound workflow transformers")
        return self._workflow_backend.result

    @property
    def status(self) -> str:
        """Return the lifecycle status of a bound workflow transformer."""

        if self._workflow_backend is None:
            raise AttributeError(
                "status is only available for bound workflow transformers"
            )
        return self._workflow_backend.status

    @property
    def exception(self):
        """Return the exception associated with a failed workflow transformer."""

        if self._workflow_backend is None:
            raise AttributeError(
                "exception is only available for bound workflow transformers"
            )
        return self._workflow_backend.exception

    def compute(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.compute()
        return self().compute()

    def run(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.run()
        return self().run()

    def task(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.task()
        return self().task()

    def prune(self):
        if self._workflow_backend is None:
            raise AttributeError("prune is only available for bound workflow transformers")
        return self._workflow_backend.prune()

    def clear_exception(self):
        if self._workflow_backend is None:
            raise AttributeError(
                "clear_exception is only available for bound workflow transformers"
            )
        return self._workflow_backend.clear_exception()

    @property
    def meta(self):
        """Transformation metadata."""

        if self._workflow_backend is not None:
            return self._workflow_backend.meta
        return self._meta

    @meta.setter
    def meta(self, meta: dict):
        if getattr(self, "_workflow_backend", None) is not None:
            self._workflow_backend.meta = meta
            return
        self._meta.update(meta)
        for k in list(self._meta.keys()):
            if self._meta[k] is None and k != "local":
                self._meta.pop(k)

    @property
    def scratch(self) -> bool:
        """If True, the transformation result buffer will not be saved."""

        if self._workflow_backend is not None:
            return self._workflow_backend.scratch
        return self._scratch

    @scratch.setter
    def scratch(self, value: bool):
        if getattr(self, "_workflow_backend", None) is not None:
            self._workflow_backend.scratch = value
            return
        self._scratch = value

    @property
    def allow_input_fingertip(self) -> bool:
        """If True, inputs may be fingertipped when resolving their buffers."""

        if self._workflow_backend is not None:
            return self._workflow_backend.allow_input_fingertip
        return bool(self._meta.get("allow_input_fingertip", False))

    @allow_input_fingertip.setter
    def allow_input_fingertip(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(type(value))
        if self._workflow_backend is not None:
            self._workflow_backend.allow_input_fingertip = value
            return
        if value:
            self.meta = {"allow_input_fingertip": True}
        else:
            self._meta.pop("allow_input_fingertip", None)

    @property
    def direct_print(self):
        """Print stdout/stderr directly instead of only storing logs."""

        if self._workflow_backend is not None:
            return self._workflow_backend.direct_print
        return self._meta.get("__direct_print__", False)

    @direct_print.setter
    def direct_print(self, value):
        if getattr(self, "_workflow_backend", None) is not None:
            self._workflow_backend.direct_print = value
            return
        if not isinstance(value, bool) and value is not None:
            raise TypeError(type(value))
        self.meta = {"__direct_print__": value}

    @property
    def driver(self) -> bool:
        """Marks the transformer as a driver script."""

        if self._workflow_backend is not None:
            return self._workflow_backend.driver
        return self._meta.get("driver", False)

    @driver.setter
    def driver(self, value):
        if not isinstance(value, bool) and value is not None:
            raise TypeError(type(value))
        if self._workflow_backend is not None:
            self._workflow_backend.driver = value
            return
        self.meta = {"driver": value}

    @property
    def local(self) -> bool | None:
        """Local execution preference."""

        if self._workflow_backend is not None:
            return self._workflow_backend.local
        return self.meta.get("local")

    @local.setter
    def local(self, value: bool | None):
        if getattr(self, "_workflow_backend", None) is not None:
            self._workflow_backend.local = value
            return
        self.meta["local"] = value

    def _workflow_endpoint(self):
        backend = self._workflow_backend
        return backend._workflow_endpoint() if backend is not None else None

    def _workflow_capture_source(self):
        backend = self._workflow_backend
        if backend is None:
            return self
        return backend.capture_source()

    # Input pins are reached through `.pins` (or its `.args` alias) only.  There is
    # deliberately no attribute or item pin sugar and no `__getattr__` fallback:
    # every Transformer attribute name is configuration API, so a pin can never be
    # shadowed by a class name such as `scratch`, `local`, `code` or `result`, and a
    # bound-only property keeps its own error instead of decaying into a pin read.

    def __setattr__(self, name, value):
        if name.startswith("_") or _class_attribute(type(self), name) is not None:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(_no_such_attribute(self, name))

    def __delattr__(self, name):
        if name.startswith("_") or _class_attribute(type(self), name) is not None:
            object.__delattr__(self, name)
            return
        raise AttributeError(_no_such_attribute(self, name))

    def _replace_checksum_field(self, old, new) -> None:
        from seamless import Checksum

        old_checksum = old if isinstance(old, Checksum) else None
        new_checksum = new if isinstance(new, Checksum) else None
        if new_checksum is not None:
            new_checksum.incref_refholder()
        if old_checksum is not None:
            old_checksum.decref_refholder()

    def _refheld_checksums(self):
        from seamless import Checksum

        if getattr(self, "_refholds_released", False):
            return ()
        claims = []
        for name, value in self._args.items():
            if isinstance(value, Checksum):
                claims.append((value, f"pin:{name}"))
        if isinstance(getattr(self, "_codebuf", None), Checksum):
            claims.append((self._codebuf, "code"))
        for name, value in self._modules.items():
            if isinstance(value, Checksum):
                claims.append((value, f"module:{name}"))
        return claims

    def _release_refholds(self) -> None:
        if getattr(self, "_refholds_released", False):
            return
        object.__setattr__(self, "_refholds_released", True)
        from seamless import Checksum
        for value in list(self._args.values()):
            if isinstance(value, Checksum):
                value.decref_refholder()
        for value in list(self._modules.values()):
            if isinstance(value, Checksum):
                value.decref_refholder()
        if isinstance(getattr(self, "_codebuf", None), Checksum):
            self._codebuf.decref_refholder()

    def __del__(self):
        try:
            self._release_refholds()
        except Exception:
            pass


def _class_attribute(cls, name):
    for parent in cls.__mro__:
        if name in parent.__dict__:
            return parent.__dict__[name]
    return None


def _no_such_attribute(obj, name: str) -> str:
    return (
        f"'{type(obj).__name__}' object has no attribute '{name}'; "
        f"transformer input pins are reached as .pins['{name}']"
    )


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
            self._workflow_callable = code
            signature = inspect.signature(code)
            code = getsource(code)
            codebuf = Buffer(code, "python")
            self._codebuf = codebuf
            self._celltypes = {k: "mixed" for k in signature.parameters}
            self._celltypes["result"] = "mixed"
        else:
            self._workflow_callable = None
            assert isinstance(code, str)
            self._codebuf = Buffer(code, "text")
        self._signature = signature

    def _get_signature(self):
        if self._workflow_backend is not None:
            cfg = self._workflow_backend.cfg
            if callable(cfg.callable):
                return inspect.signature(cfg.callable)
            return None
        return self._signature

    def _get_codebuf(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.code
        return self._codebuf

    @property
    def code(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.code
        return self._codebuf

    @code.setter
    def code(self, code: Callable[P, R] | str):
        if getattr(self, "_workflow_backend", None) is not None:
            self._workflow_backend.code = code
            return
        return self._set_code(code)

    @property
    def language(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.language
        return self._language

    @language.setter
    def language(self, lang):
        if getattr(self, "_workflow_backend", None) is not None:
            self._workflow_backend.language = lang
            return
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

    def __init__(self, owner, args, celltypes, fixed):
        self._owner = owner
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
        old = self._args.get(key)
        self._owner._replace_checksum_field(old, value)
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

    def __init__(self, owner, modules):
        self._owner = owner
        self._modules = modules

    def __getattr__(self, attr):
        return self._modules[attr]

    def __getitem__(self, key):
        return self._modules[key]

    def __setattr__(self, attr, value):
        if attr.startswith("_"):
            return super().__setattr__(attr, value)
        return self.__setitem__(attr, value)

    def __setitem__(self, key, value):
        if not isinstance(value, (ModuleType, dict)):
            raise TypeError(type(value))
        old = self._modules.get(key)
        self._owner._replace_checksum_field(old, value)
        self._modules[key] = value

    def __delattr__(self, attr: str) -> None:
        if attr.startswith("_"):
            return super().__delattr__(attr)
        return self.__delitem__(attr)

    def __delitem__(self, key) -> None:
        old = self._modules.pop(key, None)
        self._owner._replace_checksum_field(old, None)

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
