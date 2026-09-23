"""Compiled-language transformers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import inspect
from pathlib import Path
from typing import Any

import yaml

from seamless import Buffer, Checksum, ensure_open

from .environment import Environment
from .pretransformation import compiled_transformer_to_pretransformation
from .transformation_class import Transformation, transformation_from_pretransformation
from .transformer_class import ArgsWrapper, DirectCallMixin, TransformerCore


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


def _checksum_hex(value, celltype: str) -> str:
    buffer = Buffer(value, celltype)
    checksum = buffer.get_checksum()
    buffer.tempref()
    return checksum.hex()


def _validate_derived_compiled_dunders(prepared_transformation, *, header: str) -> None:
    """Validate caller-supplied derived compiled support before checksuming.

    ``__compiled__`` and ``__header__`` are derived/eliminable: they are carried
    for execution compatibility but excluded from transformation identity. If a
    caller or legacy path supplied them, they must agree with the schema-derived
    compiled state before they are discarded from the immutable definition.
    """

    supplied_compiled = prepared_transformation.get("__compiled__")
    if supplied_compiled is not None and supplied_compiled is not True:
        raise ValueError(
            "Compiled transformation supplied __compiled__ must be True"
        )

    supplied_header = prepared_transformation.get("__header__")
    if supplied_header is None:
        return
    expected_header = _checksum_hex(header, "text")
    try:
        supplied_header = Checksum(supplied_header).hex()
    except Exception as exc:
        raise ValueError(
            "Compiled transformation supplied __header__ is not a valid checksum"
        ) from exc
    if supplied_header != expected_header:
        raise ValueError(
            "Compiled transformation supplied __header__ does not match the "
            "schema-generated header"
        )


def _deferred_validation_hooks(signature):
    """Check deferred checksum facts without materializing input data."""
    from .compiled_validation import validate_prepared

    def _run_validations_sync(prepared_transformation):
        validate_prepared(signature, prepared_transformation)

    async def _run_validations_async(prepared_transformation):
        validate_prepared(signature, prepared_transformation)

    return _run_validations_sync, _run_validations_async


def _compose_post_prepare_hooks(*sync_hooks):
    hooks = [hook for hook in sync_hooks if hook is not None]
    if not hooks:
        return None

    def _run(prepared_transformation):
        for hook in hooks:
            hook(prepared_transformation)

    return _run


def _compose_post_prepare_async_hooks(always_sync_hook, async_hook):
    if always_sync_hook is None and async_hook is None:
        return None

    async def _run(prepared_transformation):
        if always_sync_hook is not None:
            always_sync_hook(prepared_transformation)
        if async_hook is not None:
            await async_hook(prepared_transformation)

    return _run


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


class BoundMetaVars(MetaVars):
    """Output limits stored on a bound transformer's canonical configuration."""

    def __init__(self, backend):
        super().__init__()
        self._backend = backend
        self._refresh()

    def _refresh(self):
        cfg = self._backend.cfg
        sig = _require_signature_package().Signature.from_dict(yaml.safe_load(cfg.schema))
        self._values = dict(cfg.meta.get("metavars", {}))
        self._rebuild(sig.output_wildcards)

    def __getattr__(self, name):
        self._refresh()
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            return super().__setattr__(name, value)
        self._refresh()
        super().__setattr__(name, value)
        backend = self._backend
        backend.context._set_node_config(backend.node_path, "meta",
                                         {"metavars": dict(self._values)})

    @property
    def is_complete(self):
        self._refresh()
        return super().is_complete

    def to_dict(self):
        self._refresh()
        return super().to_dict()


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
            from .transformer_class import CelltypesWrapper
            owner = self._transformer
            from .compiled_validation import ALLOWED_CELLTYPES, CompiledPinCelltypeError, validate_declarations
            if owner._schema is None or key not in {p.name for p in owner._schema.inputs}:
                raise AttributeError(key)
            if isinstance(value, type):
                value = value.__name__
            if value not in ALLOWED_CELLTYPES:
                raise CompiledPinCelltypeError(f"Compiled pin {key!r}: unsupported celltype {value!r}")
            CelltypesWrapper(owner, owner._celltypes, owner._args, fixed=True).__setitem__(key, value)
            validate_declarations(owner._schema, owner._celltypes, warn=True)
            return
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
        return list(self._transformer._celltypes)


class CompiledMixin:
    """Mixin that adds compiled-language attributes to a transformer core.

    Not for direct use. Consumed by CompiledTransformer and DirectCompiledTransformer.
    """

    def __init_compiled__(self, language: str):
        _require_signature_package()
        from seamless_transformer.languages import get_language

        lang_def = get_language(language)
        self._compiled_language = language
        self._compilation = deepcopy(lang_def.compilation)
        self._environment = Environment()
        self._schema_text = None
        self._schema = None
        self._call_signature = None
        self._code_text = None
        self._metavars = MetaVars()
        self._objects = ObjectList()

    def _validate_compiled_stage1(self):
        from .compiled_validation import validate_stage1
        validate_stage1(self._schema_text, self._celltypes, self._optional_pins,
                        self._metavars.to_dict())

    @property
    def schema_celltypes(self):
        """Read-only schema-derived types; declarations remain independent."""
        from .compiled_validation import SchemaCelltypesView
        if self._workflow_backend is not None:
            try:
                sig = _require_signature_package().Signature.from_dict(yaml.safe_load(self.schema))
            except Exception:
                return SchemaCelltypesView(None, {})
        else:
            sig = self._schema
        declarations = self._workflow_backend.cfg.celltypes if self._workflow_backend is not None else self._celltypes
        return SchemaCelltypesView(sig, declarations)

    def __repr__(self):
        from .compiled_validation import validate_declarations, CompiledPinCelltypeError
        if self._workflow_backend is not None:
            return f'<CompiledTransformer declared={self._workflow_backend.cfg.celltypes!r}, schema_celltypes={dict(self.schema_celltypes)!r}; {self._workflow_backend.exception or self._workflow_backend.state}>'
        diagnostic = ''
        if self._schema is not None:
            try:
                validate_declarations(self._schema, self._celltypes)
            except CompiledPinCelltypeError as exc:
                diagnostic = f'; incompatible: {exc}'
        return f'<CompiledTransformer declared={self._celltypes!r}, schema_celltypes={dict(self.schema_celltypes)!r}{diagnostic}>'

    @property
    def compilation(self):
        """Compiler binary, flags, and mode used to build this transformer."""
        if self._workflow_backend is not None:
            return self._workflow_backend.cfg.compilation
        return self._compilation

    @compilation.setter
    def compilation(self, value):
        if self._workflow_backend is not None:
            backend = self._workflow_backend
            backend.context._set_node_config(backend.node_path, "compilation", value)
            return
        self._compilation = value

    @property
    def language(self) -> str:
        """The compiled language name (read-only after construction)."""
        if self._workflow_backend is not None:
            return self._workflow_backend.cfg.language
        return self._compiled_language

    @language.setter
    def language(self, _value):
        raise AttributeError("compiled transformer language is read-only")

    @property
    def schema(self) -> str | None:
        """The seamless-signature schema YAML string, or None if not yet set."""
        if self._workflow_backend is not None:
            return self._workflow_backend.cfg.schema
        return self._schema_text

    @schema.setter
    def schema(self, value: str | Path):
        ss = _require_signature_package()
        if isinstance(value, Path):
            value = value.read_text()
        if not isinstance(value, str):
            raise TypeError(type(value))
        if self._workflow_backend is not None:
            backend = self._workflow_backend
            backend.context._set_node_config(backend.node_path, 'schema', value)
            return
        data = yaml.safe_load(value)
        sig = ss.Signature.from_dict(data)
        self._validate_schema(sig)
        ss.generate_header(sig)
        names = {p.name for p in sig.inputs}
        for name in set(self._args) - names:
            old, _ = self._args.pop(name)
            self._replace_checksum_field(old, None)
            self._pin_memos.pop(name, None)
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
        self._celltypes = {parameter.name: self._celltypes.get(parameter.name, "mixed") for parameter in sig.inputs}
        self._celltypes["result"] = result_celltype
        from .compiled_validation import validate_declarations
        validate_declarations(sig, self._celltypes, warn=True)
        if len(sig.outputs) > 1 and self._celltypes["result"] not in ("mixed", "deepcell"):
            raise TypeError("multi-output compiled transformers require result celltype 'mixed' or 'deepcell'")

    def _validate_schema(self, sig):
        return None

    @property
    def code(self) -> str | None:
        """The compiled source code string, or None if not yet set.

        Accepts a string or a pathlib.Path (file contents are read immediately).
        """
        if self._workflow_backend is not None:
            return self._workflow_backend.code
        return self._code_text

    @code.setter
    def code(self, value: str | Path):
        if isinstance(value, Path):
            value = value.read_text()
        if not isinstance(value, str):
            raise TypeError(type(value))
        if self._workflow_backend is not None:
            self._workflow_backend.code = value
            return
        self._code_text = value

    @property
    def header(self) -> str | None:
        """C header generated from the current schema, or None if schema is not set.

        Generated by seamless-signature from the schema YAML. This is the header
        that defines the ``transform()`` function signature in C, and is also
        passed to CFFI to build the Python extension module.
        """
        if self._workflow_backend is not None:
            return self._workflow_backend.cfg.header
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
        if self._workflow_backend is not None:
            return BoundMetaVars(self._workflow_backend)
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
        if self._workflow_backend is not None:
            return self._workflow_backend.celltypes
        return CompiledCelltypesWrapper(self)

    @property
    def args(self):
        """Pre-bound input arguments, same as for Python transformers."""
        if self._workflow_backend is not None:
            return self._workflow_backend.args
        return ArgsWrapper(self, self._args, self._celltypes, fixed=self._call_signature is not None)

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

    def _snapshot_for_call(self):
        if self._workflow_backend is not None:
            return self._workflow_backend.snapshot_for_call()
        from .builder_snapshot import TransformerBuilderSnapshot
        from seamless import Buffer
        objects, compilation = self._compiled_payloads()
        meta = deepcopy(self._meta)
        meta.setdefault("metavars", self._metavars.to_dict())
        pin_args, input_celltypes = self._snapshot_pin_inputs()
        return TransformerBuilderSnapshot(
            codebuf=Buffer(self._code_text, "text") if self._code_text is not None else None,
            language=self.language, celltypes=deepcopy(self._celltypes),
            optional_pins=frozenset(self._optional_pins), args=pin_args,
            input_celltypes=input_celltypes,
            modules={}, globals={}, meta=meta, environment=self._environment._to_lowlevel(),
            scratch=self.scratch, direct_print=self.direct_print, local=self.local,
            call_mode="direct" if isinstance(self, DirectCallMixin) else "delayed",
            signature=self._call_signature,
            schema=self._schema_text, compilation=compilation, objects=objects, header=self.header)

    def _bind_compiled_arguments(self, *args, **kwargs):
        self._validate_compiled_stage1()
        if self._code_text is None:
            raise ValueError("compiled transformer code is not set")
        all_args = {name: self.pins[name].build() for name in self._args}
        all_args.update(self._call_signature.bind_partial(*args, **kwargs).arguments)
        arguments = self._call_signature.bind(**all_args).arguments
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
    """Delayed compiled-language transformer.

    Wraps C, C++, Fortran, or Rust source code as a Seamless transformation.
    Calling the transformer returns a :class:`Transformation` handle that can
    be executed later — the same delayed semantics as :func:`delayed` for
    Python transformers.

    Basic usage::

        from seamless_transformer import Transformer

        tf = Transformer("c", compiled=True, direct=True)
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

        tf = Transformer("c", compiled=True)
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
        if self._workflow_backend is not None:
            return self._build_from_snapshot(self._snapshot_for_call(), *args, **kwargs)
        if self._modules or self._globals:
            raise NotImplementedError("modules/globals are not supported for compiled transformers")
        arguments = self._bind_compiled_arguments(*args, **kwargs)
        self._convert_pin_arguments(arguments, self._celltypes)
        from seamless import Expression
        deps = {
            argname: arg
            for argname, arg in arguments.items()
            if isinstance(arg, (Transformation, Expression))
        }
        meta = deepcopy(self._meta)
        meta.setdefault("metavars", self._metavars.to_dict())
        objects, compilation = self._compiled_payloads()
        header = self.header
        pre_transformation = compiled_transformer_to_pretransformation(
            code=self._code_text,
            schema_text=self._schema_text,
            header=header,
            compilation=compilation,
            objects=objects,
            meta=meta,
            celltypes=self._celltypes,
            arguments=arguments,
            env=self._environment._to_lowlevel(),
            language=self.language,
            optional_pins=self._optional_pins,
        )
        deferred_prepare_sync, deferred_prepare_async = _deferred_validation_hooks(self._schema)
        derived_dunder_validation = lambda prepared: _validate_derived_compiled_dunders(
            prepared, header=header
        )
        post_prepare_sync = _compose_post_prepare_hooks(
            derived_dunder_validation, deferred_prepare_sync
        )
        post_prepare_async = _compose_post_prepare_async_hooks(
            derived_dunder_validation, deferred_prepare_async
        )
        tf = transformation_from_pretransformation(
            pre_transformation,
            upstream_dependencies=deps,
            meta=meta,
            scratch=self.scratch,
            tf_dunder={},
            post_prepare_sync=post_prepare_sync,
            post_prepare_async=post_prepare_async,
        )
        return tf


class DirectCompiledTransformer(DirectCallMixin, CompiledTransformer):
    """Compiled transformer that computes immediately and returns the value.

    Identical to :class:`CompiledTransformer` except that calling the
    transformer runs the compilation and execution pipeline immediately and
    returns the result value, rather than a :class:`Transformation` handle.

    Use this for interactive or script workflows where you want an immediate
    result. For pipeline or deferred execution, use :class:`CompiledTransformer`.
    """

    def _direct_result(self, transformation):
        value = transformation.run()
        if transformation.celltype == "deepcell":
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
