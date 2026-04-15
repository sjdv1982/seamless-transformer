"""Compiled-language transformers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import inspect
from pathlib import Path
from typing import Any

import yaml

from seamless import ensure_open

from .environment import Environment
from .pretransformation import compiled_transformer_to_pretransformation
from .transformation_class import Transformation, transformation_from_pretransformation
from .transformer_class import ArgsWrapper, TransformerCore


def _require_signature_package():
    try:
        import seamless_signature
    except ImportError:
        raise ImportError(
            "seamless-signature is required for compiled transformers. "
            "Install it with: pip install seamless-signature"
        ) from None
    return seamless_signature


def _as_plain(value):
    if is_dataclass(value):
        return asdict(value)
    return deepcopy(value)


def _expected_numpy_dtype(dtype_name: str):
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required for compiled transformer array/scalar validation. "
            "Install it with: pip install numpy"
        ) from None

    if dtype_name == "char":
        return np.dtype("S1")
    return np.dtype(dtype_name)


def _validate_native_numpy_value(name: str, value, dtype_name: str, is_array: bool):
    try:
        import numpy as np
    except ImportError:
        if is_array:
            raise ImportError(
                "numpy is required for compiled transformer array inputs. "
                "Install it with: pip install numpy"
            ) from None
        return

    expected = _expected_numpy_dtype(dtype_name)
    if is_array:
        array = np.asarray(value)
        if array.dtype != expected:
            raise TypeError(f"Input {name!r} dtype is {array.dtype}, expected {expected}")
        if array.dtype.byteorder not in ("=", "|"):
            raise TypeError(f"Input {name!r} must have native byte order")
    elif isinstance(value, np.generic):
        if value.dtype != expected:
            raise TypeError(f"Input {name!r} dtype is {value.dtype}, expected {expected}")
        if value.dtype.byteorder not in ("=", "|"):
            raise TypeError(f"Input {name!r} must have native byte order")


class MetaVars:
    """Dynamic attribute namespace for output-wildcard max-values."""

    def __init__(self):
        self._allowed: set[str] = set()
        self._values: dict[str, int] = {}

    def _rebuild(self, output_wildcards: tuple[str, ...]):
        new_allowed = {f"max{w}" for w in output_wildcards}
        for key in list(self._values):
            if key not in new_allowed:
                del self._values[key]
        self._allowed = new_allowed

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name not in self._allowed:
            raise AttributeError(f"No metavar {name!r}")
        return self._values.get(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            return super().__setattr__(name, value)
        if name not in self._allowed:
            raise AttributeError(f"No metavar {name!r}")
        self._values[name] = int(value)

    @property
    def is_complete(self) -> bool:
        return all(name in self._values for name in self._allowed)

    def to_dict(self) -> dict[str, int]:
        return dict(self._values)


class CompiledObject:
    """Additional compiled source object linked into a compiled transformer."""

    def __init__(self, *, language: str):
        from seamless_transformer.languages import get_language

        lang_def = get_language(language)
        self._language = language
        self.compilation = deepcopy(lang_def.compilation)
        self._code = None

    @property
    def language(self) -> str:
        return self._language

    @property
    def code(self) -> str | None:
        return self._code

    @code.setter
    def code(self, value: str | Path):
        if isinstance(value, Path):
            value = value.read_text()
        if not isinstance(value, str):
            raise TypeError(type(value))
        self._code = value

    def to_object_payload(self, name: str) -> dict[str, Any]:
        if self._code is None:
            raise ValueError(f"Compiled object {name!r} has no code")
        return {"name": name, "language": self.language, "code": self._code}

    def to_compilation_payload(self) -> dict[str, Any]:
        return _as_plain(self.compilation)


class ObjectList:
    """List-like container of CompiledObject instances."""

    def __init__(self):
        self._objects: list[CompiledObject] = []

    def append(self, obj: CompiledObject):
        if not isinstance(obj, CompiledObject):
            raise TypeError(type(obj))
        self._objects.append(obj)

    def __getitem__(self, index):
        return self._objects[index]

    def __len__(self):
        return len(self._objects)

    def __iter__(self):
        return iter(self._objects)


class CompiledCelltypesWrapper:
    """Restricted celltype wrapper for compiled transformers."""

    def __init__(self, transformer: "CompiledMixin"):
        self._transformer = transformer

    def __getattr__(self, attr):
        return self._transformer._celltypes[attr]

    def __getitem__(self, key):
        return self._transformer._celltypes[key]

    def __setattr__(self, attr, value):
        if attr.startswith("_"):
            return super().__setattr__(attr, value)
        return self.__setitem__(attr, value)

    def __setitem__(self, key, value):
        if key != "result":
            raise AttributeError(key)
        if isinstance(value, type):
            value = value.__name__
        value = str(value)
        if self._transformer._schema is not None and len(self._transformer._schema.outputs) > 1:
            if value not in ("mixed", "deepcell"):
                raise TypeError("multi-output compiled transformers require result celltype 'mixed' or 'deepcell'")
        elif value not in ("mixed", "deepcell"):
            raise TypeError("compiled transformer result celltype must be 'mixed' or 'deepcell'")
        self._transformer._celltypes["result"] = value

    def __dir__(self):
        return ["result"]


class CompiledMixin:
    """Mixin that adds compiled-language attributes to a transformer core."""

    def __init_compiled__(self, language: str):
        _require_signature_package()
        from seamless_transformer.languages import get_language

        lang_def = get_language(language)
        self._compiled_language = language
        self.compilation = deepcopy(lang_def.compilation)
        self._environment = Environment()
        self._schema_text = None
        self._schema = None
        self._call_signature = None
        self._code_text = None
        self._metavars = MetaVars()
        self._objects = ObjectList()

    @property
    def language(self) -> str:
        return self._compiled_language

    @language.setter
    def language(self, _value):
        raise AttributeError("compiled transformer language is read-only")

    @property
    def schema(self) -> str | None:
        return self._schema_text

    @schema.setter
    def schema(self, value: str | Path):
        ss = _require_signature_package()
        if isinstance(value, Path):
            value = value.read_text()
        if not isinstance(value, str):
            raise TypeError(type(value))
        data = yaml.safe_load(value)
        sig = ss.Signature.from_dict(data)
        self._validate_schema(sig)
        self._schema_text = value
        self._schema = sig
        self._metavars._rebuild(sig.output_wildcards)
        self._call_signature = inspect.Signature(
            [
                inspect.Parameter(
                    parameter.name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for parameter in sig.inputs
            ]
        )
        result_celltype = self._celltypes.get("result", "mixed")
        self._celltypes = {parameter.name: "mixed" for parameter in sig.inputs}
        self._celltypes["result"] = result_celltype
        if len(sig.outputs) > 1 and self._celltypes["result"] not in ("mixed", "deepcell"):
            raise TypeError("multi-output compiled transformers require result celltype 'mixed' or 'deepcell'")

    def _validate_schema(self, sig):
        ss = _require_signature_package()
        for parameter in sig.inputs + sig.outputs:
            if isinstance(parameter.dtype, ss.StructDType):
                raise NotImplementedError("Structured dtypes are not supported yet")

    @property
    def code(self) -> str | None:
        return self._code_text

    @code.setter
    def code(self, value: str | Path):
        if isinstance(value, Path):
            value = value.read_text()
        if not isinstance(value, str):
            raise TypeError(type(value))
        self._code_text = value

    @property
    def header(self) -> str | None:
        if self._schema is None:
            return None
        ss = _require_signature_package()
        return ss.generate_header(self._schema)

    @property
    def metavars(self) -> MetaVars:
        return self._metavars

    @property
    def objects(self) -> ObjectList:
        return self._objects

    @property
    def celltypes(self):
        return CompiledCelltypesWrapper(self)

    @property
    def args(self):
        return ArgsWrapper(self._args, self._celltypes, fixed=self._call_signature is not None)

    @property
    def modules(self):
        raise NotImplementedError("modules are not supported for compiled transformers")

    @property
    def globals(self):
        raise NotImplementedError("globals are not supported for compiled transformers")

    def _get_signature(self):
        return self._call_signature

    def _bind_compiled_arguments(self, *args, **kwargs):
        if self._schema is None:
            raise ValueError("compiled transformer schema is not set")
        if self._code_text is None:
            raise ValueError("compiled transformer code is not set")
        if not self._metavars.is_complete:
            missing = sorted(self._metavars._allowed - self._metavars._values.keys())
            raise ValueError(f"compiled transformer metavars are incomplete: {missing}")
        all_args = self._args.copy()
        all_args.update(self._call_signature.bind_partial(*args, **kwargs).arguments)
        arguments = self._call_signature.bind(**all_args).arguments
        for parameter in self._schema.inputs:
            _validate_native_numpy_value(
                parameter.name,
                arguments[parameter.name],
                parameter.dtype.name,
                parameter.shape is not None,
            )
        return arguments

    def _compiled_payloads(self):
        objects = {}
        object_compilations = {}
        for index, obj in enumerate(self._objects):
            name = f"obj{index}"
            payload = obj.to_object_payload(name)
            objects[name] = payload
            object_compilations[name] = obj.to_compilation_payload()
        compilation = {
            "main": _as_plain(self.compilation),
            "objects": object_compilations,
        }
        return objects, compilation


class CompiledTransformer(CompiledMixin, TransformerCore):
    """Delayed compiled-language transformer."""

    def __init__(
        self,
        language: str,
        *,
        scratch: bool = False,
        direct_print: bool = False,
        local: bool = False,
    ):
        self._init_core(
            language=language,
            scratch=scratch,
            direct_print=direct_print,
            local=local,
        )
        self._celltypes = {"result": "mixed"}
        self.__init_compiled__(language)

    def __call__(self, *args, **kwargs) -> Transformation:
        ensure_open("compiled transformer call")
        if self._modules or self._globals:
            raise NotImplementedError("modules/globals are not supported for compiled transformers")
        arguments = self._bind_compiled_arguments(*args, **kwargs)
        deps = {
            argname: arg
            for argname, arg in arguments.items()
            if isinstance(arg, Transformation)
        }
        meta = deepcopy(self._meta)
        meta.setdefault("metavars", self._metavars.to_dict())
        objects, compilation = self._compiled_payloads()
        pre_transformation = compiled_transformer_to_pretransformation(
            code=self._code_text,
            schema_text=self._schema_text,
            header=self.header,
            compilation=compilation,
            objects=objects,
            meta=meta,
            celltypes=self._celltypes,
            arguments=arguments,
            env=self._environment._to_lowlevel(),
            language=self.language,
        )
        return transformation_from_pretransformation(
            pre_transformation,
            upstream_dependencies=deps,
            meta=meta,
            scratch=self.scratch,
            tf_dunder={},
        )


class DirectCompiledTransformer(CompiledTransformer):
    """Compiled transformer that computes immediately."""

    def __call__(self, *args, **kwargs):
        tf = super().__call__(*args, **kwargs)
        tf._compute(api_origin="call")
        value = tf.run()
        if tf.celltype == "deepcell":
            from .transformation_utils import unpack_deep_structure

            return unpack_deep_structure(value, "deepcell")
        return value


__all__ = [
    "CompiledObject",
    "CompiledTransformer",
    "DirectCompiledTransformer",
    "MetaVars",
    "ObjectList",
]
