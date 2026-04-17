"""Compiled-language transformers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import inspect
from pathlib import Path
from typing import Any

import yaml

from seamless import Checksum, ensure_open

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


def _expected_numpy_dtype(dtype_spec):
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required for compiled transformer array/scalar validation. "
            "Install it with: pip install numpy"
        ) from None

    if hasattr(dtype_spec, "fields"):
        fields = []
        for field in dtype_spec.fields:
            field_dtype = _expected_numpy_dtype(field.dtype)
            if field.shape:
                fields.append((field.name, field_dtype, field.shape))
            else:
                fields.append((field.name, field_dtype))
        return np.dtype(fields, align=True)

    dtype_name = dtype_spec.name
    if dtype_name == "char":
        return np.dtype("S1")
    return np.dtype(dtype_name)


def _numpy_dtype_matches(actual, expected) -> bool:
    if actual == expected:
        return True
    if actual.fields is None or expected.fields is None:
        return False
    if actual.itemsize != expected.itemsize:
        return False

    expected_names = set(expected.names or ())
    for name in actual.names or ():
        if name in expected_names:
            continue
        field_dtype = actual.fields[name][0]
        if field_dtype.kind != "V" or field_dtype.fields is not None:
            return False

    for name in expected.names or ():
        if name not in actual.fields:
            return False
        actual_field, actual_offset = actual.fields[name][:2]
        expected_field, expected_offset = expected.fields[name][:2]
        if actual_offset != expected_offset:
            return False
        actual_base, actual_shape = actual_field.subdtype or (actual_field, ())
        expected_base, expected_shape = expected_field.subdtype or (expected_field, ())
        if actual_shape != expected_shape:
            return False
        if not _numpy_dtype_matches(actual_base, expected_base):
            return False
    return True


def _validate_native_numpy_value(name: str, value, dtype_spec, is_array: bool):
    try:
        import numpy as np
    except ImportError:
        if is_array:
            raise ImportError(
                "numpy is required for compiled transformer array inputs. "
                "Install it with: pip install numpy"
            ) from None
        return

    expected = _expected_numpy_dtype(dtype_spec)
    if is_array:
        array = np.asarray(value)
        if not _numpy_dtype_matches(array.dtype, expected):
            raise TypeError(f"Input {name!r} dtype is {array.dtype}, expected {expected}")
        if not array.dtype.isnative:
            raise TypeError(f"Input {name!r} must have native byte order")
    elif hasattr(dtype_spec, "fields"):
        if isinstance(value, np.ndarray):
            if value.shape != ():
                raise TypeError(f"Input {name!r} must be a scalar structured value")
            dtype = value.dtype
        elif isinstance(value, np.void):
            dtype = value.dtype
        else:
            raise TypeError(
                f"Input {name!r} must be a numpy structured scalar with dtype {expected}"
            )
        if not _numpy_dtype_matches(dtype, expected):
            raise TypeError(f"Input {name!r} dtype is {dtype}, expected {expected}")
        if not dtype.isnative:
            raise TypeError(f"Input {name!r} must have native byte order")
    elif isinstance(value, np.generic):
        if value.dtype != expected:
            raise TypeError(f"Input {name!r} dtype is {value.dtype}, expected {expected}")
        if not value.dtype.isnative:
            raise TypeError(f"Input {name!r} must have native byte order")


def _is_deferred_input(value) -> bool:
    """Return True if the input must be resolved before dtype validation."""
    if isinstance(value, (Checksum, Transformation)):
        return True
    if isinstance(value, str) and len(value) == 64:
        try:
            int(value, 16)
        except ValueError:
            return False
        return True
    if isinstance(value, (bytes, bytearray)) and len(value) == 32:
        return True
    return False


def _resolve_deferred_value(prepared_value):
    """Resolve a prepared pin value (hex string or None) to a Python value."""
    if prepared_value is None:
        return None
    checksum = Checksum(prepared_value)
    return checksum.fingertip_sync("mixed")


async def _resolve_deferred_value_async(prepared_value):
    if prepared_value is None:
        return None
    checksum = Checksum(prepared_value)
    return await checksum.fingertip("mixed")


def _install_deferred_validation(
    tf: Transformation,
    pre_transformation,
    deferred_validations: "list[tuple[str, Any, bool]]",
) -> None:
    """Run _validate_native_numpy_value after deferred inputs are materialized.

    The checks run after prepare_transformation has replaced Transformation/
    Checksum pins with concrete hex checksums, so the resolved buffer carries
    the value actually supplied to the compiled runner.
    """

    original_sync = tf._constructor_sync
    original_async = tf._constructor_async
    pretransformation_dict = pre_transformation.pretransformation_dict

    def _run_validations_sync():
        for name, dtype_spec, is_array in deferred_validations:
            pin = pretransformation_dict.get(name)
            if pin is None:
                continue
            prepared_value = pin[2]
            value = _resolve_deferred_value(prepared_value)
            _validate_native_numpy_value(name, value, dtype_spec, is_array)

    async def _run_validations_async():
        for name, dtype_spec, is_array in deferred_validations:
            pin = pretransformation_dict.get(name)
            if pin is None:
                continue
            prepared_value = pin[2]
            value = await _resolve_deferred_value_async(prepared_value)
            _validate_native_numpy_value(name, value, dtype_spec, is_array)

    def wrapped_sync(transformation_obj):
        result = original_sync(transformation_obj)
        _run_validations_sync()
        return result

    async def wrapped_async(transformation_obj):
        result = await original_async(transformation_obj)
        await _run_validations_async()
        return result

    tf._constructor_sync = wrapped_sync
    tf._constructor_async = wrapped_async


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
    """Mixin that adds compiled-language attributes to a transformer core.

    Not for direct use. Consumed by CompiledTransformer and DirectCompiledTransformer.
    """

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
        """The compiled language name (read-only after construction)."""
        return self._compiled_language

    @language.setter
    def language(self, _value):
        raise AttributeError("compiled transformer language is read-only")

    @property
    def schema(self) -> str | None:
        """The seamless-signature schema YAML string, or None if not yet set."""
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
        return None

    @property
    def code(self) -> str | None:
        """The compiled source code string, or None if not yet set.

        Accepts a string or a pathlib.Path (file contents are read immediately).
        """
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
        """C header generated from the current schema, or None if schema is not set.

        Generated by seamless-signature from the schema YAML. This is the header
        that defines the ``transform()`` function signature in C, and is also
        passed to CFFI to build the Python extension module.
        """
        if self._schema is None:
            return None
        ss = _require_signature_package()
        return ss.generate_header(self._schema)

    @property
    def metavars(self) -> MetaVars:
        """Output-wildcard max-value namespace.

        If the schema has output-only wildcard dimensions (e.g. ``K``), this
        namespace exposes ``.maxK`` for user assignment before calling the
        transformer. These values bound the allocated output buffer size; the
        actual runtime output size is reported back by the compiled function.

        Changing the schema rebuilds metavars, dropping any stale entries.
        """
        return self._metavars

    @property
    def objects(self) -> ObjectList:
        """Additional compiled objects linked into this transformer.

        Each entry is a CompiledObject with its own language and source code.
        Objects may use a different language from the main transformer (e.g.
        a Fortran helper linked into a C transformer).
        """
        return self._objects

    @property
    def celltypes(self):
        return CompiledCelltypesWrapper(self)

    @property
    def args(self):
        """Pre-bound input arguments, same as for Python transformers."""
        return ArgsWrapper(self._args, self._celltypes, fixed=self._call_signature is not None)

    @property
    def modules(self):
        """Not supported for compiled transformers. Raises NotImplementedError if accessed."""
        raise NotImplementedError("modules are not supported for compiled transformers")

    @property
    def globals(self):
        """Not supported for compiled transformers. Raises NotImplementedError if accessed."""
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
        deferred_validations: list[tuple[str, Any, bool]] = []
        for parameter in self._schema.inputs:
            value = arguments[parameter.name]
            is_array = parameter.shape is not None
            if _is_deferred_input(value):
                deferred_validations.append((parameter.name, parameter.dtype, is_array))
                continue
            _validate_native_numpy_value(
                parameter.name,
                value,
                parameter.dtype,
                is_array,
            )
        return arguments, deferred_validations

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
    """Delayed compiled-language transformer.

    Wraps C, C++, Fortran, or Rust source code as a Seamless transformation.
    Calling the transformer returns a :class:`Transformation` handle that can
    be executed later — the same delayed semantics as :func:`delayed` for
    Python transformers.

    Basic usage::

        from seamless_transformer import CompiledTransformer, DirectCompiledTransformer

        tf = DirectCompiledTransformer("c")
        tf.schema = \"\"\"
        inputs:
          - name: a
            dtype: int32
          - name: b
            dtype: int32
        outputs:
          - name: result
            dtype: int32
        \"\"\"
        tf.code = \"\"\"
        #include <stdint.h>
        int transform(int32_t a, int32_t b, int32_t *result) {
            *result = a + b;
            return 0;
        }
        \"\"\"
        assert tf(a=2, b=3) == 5

    For a delayed (non-direct) workflow::

        tf = CompiledTransformer("c")
        tf.schema = ...
        tf.code = ...
        t = tf(a=2, b=3)    # returns a Transformation
        value = t.run()     # execute and return value

    **Schema**: a YAML string in the seamless-signature format. Accepts a
    string or a :class:`pathlib.Path`. The schema defines input/output parameter
    names, dtypes, and shapes. Struct parameters map to aligned NumPy
    structured dtypes and generated C structs in the header.

    **Code**: the compiled source as a string or :class:`pathlib.Path`. The
    source must define a ``transform()`` function matching the schema signature
    generated in ``tf.header``.

    **Compilation settings**: ``tf.compilation`` is a
    :class:`~seamless_transformer.languages.CompilationConfig` dataclass with
    the compiler binary, flags, and mode. Modify it before calling to override
    defaults (e.g. switch from ``profile`` to ``debug``).

    **Environment**: ``tf.environment`` is an :class:`~seamless_transformer.environment.Environment`
    that controls conda, docker, and execution powers for the transformation.

    **Additional objects**: ``tf.objects`` holds a list of
    :class:`CompiledObject` instances — extra source files in the same or a
    different compiled language (e.g. a Fortran helper) that are compiled and
    linked alongside the main source.

    **Caching**: transformation identity is determined by source code content
    and input values, not by compiler flags. Two runs with the same code and
    inputs but different ``-O`` flags share a cache entry. This matches
    Seamless's content-addressed identity model.

    Constructor arguments:

    - ``language``: registered compiled language name (``"c"``, ``"cpp"``,
      ``"fortran"``, ``"rust"``, or any custom language added via
      :func:`~seamless_transformer.languages.define_compiled_language`).
    - ``scratch``: if True, the result buffer may be dropped after computation.
    - ``direct_print``: if True, forward transformer stdout/stderr directly.
    - ``local``: if True, force local execution (ignore remote backend config).

    Requires ``seamless-signature``, ``cffi``, and ``numpy`` (install with
    ``pip install seamless-transformer[compiled]``).
    """

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
        arguments, deferred_validations = self._bind_compiled_arguments(*args, **kwargs)
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
        tf = transformation_from_pretransformation(
            pre_transformation,
            upstream_dependencies=deps,
            meta=meta,
            scratch=self.scratch,
            tf_dunder={},
        )
        if deferred_validations:
            _install_deferred_validation(tf, pre_transformation, deferred_validations)
        return tf


class DirectCompiledTransformer(CompiledTransformer):
    """Compiled transformer that computes immediately and returns the value.

    Identical to :class:`CompiledTransformer` except that calling the
    transformer runs the compilation and execution pipeline immediately and
    returns the result value, rather than a :class:`Transformation` handle.

    Use this for interactive or script workflows where you want an immediate
    result. For pipeline or deferred execution, use :class:`CompiledTransformer`.
    """

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
