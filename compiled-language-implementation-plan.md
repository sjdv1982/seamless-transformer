# Implementation Plan: Compiled Language Support for seamless-transformer

## Overview

This plan adds compiled language support to `seamless-transformer`, covering
the full pipeline from language registration through compilation, CFFI linking,
and `.so` generation. It also ports `Environment` from legacy Seamless as a
prerequisite.

The plan is organized into eight steps. Step 0 is a prerequisite refactor of
the transformer architecture. Steps 1–2 are prerequisites
(environment, language registry). Steps 3–6 are the core compiled transformer
data model, pretransformation support, and build pipeline. Step 7 is tests.

This document is a handoff specification, not just a roadmap. Where the
current code leaves room for interpretation, this plan defines the required
behavior explicitly so an implementation agent does not need to invent APIs or
data shapes mid-flight.

### Phase-1 scope boundary

The initial compiled-language port supports:

- A shared transformer core with Python-specific logic factored into a
    `PythonMixin`.
- `Environment` on ordinary transformers.
- Registration of compiled languages and compilation settings.
- Compiled transformers with one or more schema outputs.
- Scalar parameters and homogeneous array parameters described by
    `seamless-signature`.
- Output-only wildcards via `.metavars.maxX`.
- Additional compiled objects (`.objects`) in `"object"`/`"archive"` mode only.
- Multi-output result packaging using Seamless result celltype `"mixed"` or
  `"deepcell"`.

The initial port does **not** support:

- Structured dtypes (`StructDType`) in compiled execution. Reject them with
    `NotImplementedError` during schema validation.
- Python modules/globals injection for compiled transformers.
- Alternate compile modes beyond `"object"` and `"archive"`.
- A dedicated `BashMixin`. Bash remains on the shared transformer core path in
    phase 1; the refactor must preserve room for a future bash-specific mixin if
    later requirements justify it.

---

## Corrections applied

These corrections override the `environment-porting-plan.md` where they
conflict:

- **No `ContextEnvironment`** — only `Environment`. Language/compiler tables
  do not live on an environment class; they live in the new
  `seamless_transformer.languages` module.
- **No cson** — no `cson2json`, no `.cson` files, no `format="cson"` API.
  Language/compiler definitions are plain Python dicts.
- **Drop "extension" storage** — file extensions are not stored in language
  definitions; the compiled transformer infers them from language name or
  receives them as part of `CompiledObject`.
- **The environment plan's proposed test** (changing a Python function's
  language to `"bash"`) is invalid and will be replaced with sensible tests.

---

## Step 0 — Refactor `Transformer` Into Core + `PythonMixin`

Before adding compiled transformers, factor Python-specific behavior out of the
current `Transformer` implementation.

### Motivation

The current `Transformer` class mixes together:

- Generic transformation state and call assembly.
- Python-source handling (`_set_code`, `_codebuf`, `_signature`, callable
    wrapping).
- Language selection for Python versus bash.

Compiled transformers need the first category but not the second. Without this
refactor, compiled support has to bypass or emulate Python-only state, which is
the wrong architectural direction.

### Required refactor shape

Create a generic core class, referred to here as `TransformerCore`, with these
responsibilities:

- `_args`, `_modules`, `_globals`, `_celltypes`, `_meta`, `_scratch`,
    `direct_print`, `driver`, `local`, and `environment` state.
- The generic `__call__()` flow that turns current state plus bound arguments
    into a pretransformation and then a `Transformation`.
- Wrapper objects such as `ArgsWrapper`, `CelltypesWrapper`, `ModulesWrapper`,
    and `GlobalsWrapper`, unless a later refactor chooses to separate those too.

Create a `PythonMixin` with these responsibilities:

- `_set_code()` for Python callables and Python source text.
- `_codebuf` management.
- `_signature` management for Python callables.
- Python callable wrapping behavior currently done in `direct()` / `delayed()`.

Required resulting class relationships:

```text
TransformerCore
        ↑
PythonMixin + TransformerCore -> Transformer / DirectTransformer
CompiledMixin + TransformerCore -> CompiledTransformer / DirectCompiledTransformer
```

### Bash-specific decision

Do **not** introduce a `BashMixin` in phase 1.

Reasoning:

- Bash currently behaves as text-source execution with a runtime branch, not as
    a distinct state model with the same degree of language-specific machinery as
    Python.
- Introducing `BashMixin` now would increase refactor scope without resolving a
    concrete current problem.
- The refactor must, however, preserve a clean seam for a future `BashMixin` if
    bash later acquires distinct source/state behavior worth isolating.

### Refactor acceptance criteria

- `TransformerCore` contains no Python-specific `_signature` or `_codebuf`
    assumptions.
- Python delayed/direct transformers continue to behave exactly as before.
- Bash transformers continue to behave exactly as before.
- Compiled transformers can be built on top of `TransformerCore` without
    bypassing Python-only initialization.

---

## Step 1 — Create `seamless_transformer/environment.py`

Port `Environment` as a standalone class. This is a simplified version of
legacy `Environment` without the reactive lifecycle, without cson support,
and without `ContextEnvironment`.

### What to port

| Legacy feature | Port? | Notes |
| --- | --- | --- |
| `_props`: `_conda`, `_conda_env_name`, `_which`, `_powers`, `_docker` | Yes | All five properties |
| `_save()` / `_load()` | Yes | Serialization round-trip |
| `_to_lowlevel(*, bash=False)` | Yes | Bridge to `__env__` pipeline |
| Public setters/getters | Yes | `set_conda`, `set_conda_env`, etc. |
| `weakref` parent reference | No | No reactive sync in current design |
| `_update()` / `_sync()` lifecycle | No | Transformer reads env at call time |
| `ContextEnvironment` | No | Language tables live in `languages` module |
| `set_ipy_templates` / `set_py_bridges` | No | Out of scope |
| cson format support | No | Plain dicts only |

### Key name alignment (`conda_environment`)

The current code in `run.py` reads `env_dict.get("conda_environment", "")`.
The legacy `_to_lowlevel(bash=True)` emits `"conda_bash_env_name"`, which
does not match.

**Resolution**: `_to_lowlevel()` will emit `"conda_environment"` (matching
the current consumer in `run.py` and `cmd/api/main.py`). The legacy key names
`conda_env_name` / `conda_bash_env_name` are not carried forward — `run.py`
already handles both Python and bash paths using the single
`"conda_environment"` key.

### Class sketch

```python
# seamless_transformer/environment.py

import json
from copy import deepcopy

import yaml  # PyYAML, already a dependency


class Environment:
    """Execution environment for an individual transformer."""

    _props = ("_conda", "_conda_env_name", "_which", "_powers", "_docker")

    def __init__(self):
        self._conda = None
        self._conda_env_name = None
        self._which = None
        self._powers = None
        self._docker = None

    def _save(self) -> dict | None:
        state = {}
        for prop in self._props:
            v = getattr(self, prop)
            if v is not None:
                state[prop[1:]] = v
        return state or None

    def _load(self, state: dict):
        for prop in self._props:
            v = state.get(prop[1:])
            if v is not None:
                setattr(self, prop, v)

    def _to_lowlevel(self) -> dict | None:
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

    # --- public API ---------------------------------------------------------

    def set_conda(self, conda):
        # Accept YAML string, file path string, or file-like object
        ...

    def set_conda_env(self, conda_env_name):
        self._conda_env_name = conda_env_name

    def get_conda(self):
        return deepcopy(self._conda)

    def set_which(self, which):
        # list of binary names that must be on PATH
        ...

    def get_which(self):
        return deepcopy(self._which)

    def set_powers(self, powers):
        ...

    def get_powers(self):
        return deepcopy(self._powers)

    def set_docker(self, docker: dict):
        ...

    def get_docker(self):
        return deepcopy(self._docker)
```

### Differences from `environment-porting-plan.md`

1. No `bash` parameter on `_to_lowlevel()` — emit `"conda_environment"`
   always (matching existing consumers).
2. No `ContextEnvironment` subclass.
3. No `format` parameter on setters (no cson support).
4. Uses `yaml.safe_load` (PyYAML) instead of `ruamel.yaml`.

---

## Step 2 — Wire `Environment` into `Transformer`

### In `transformer_class.py`

**`__init__`**: replace the STUB comment with:

```python
from .environment import Environment
self._environment = Environment()
```

**Add the `environment` property**, replacing the STUB block:

```python
@property
def environment(self) -> "Environment":
    """Execution environment for this transformer."""
    return self._environment
```

**In `__call__`**, replace `env = None`:

```python
env = self._environment._to_lowlevel()
```

The existing call to `direct_transformer_to_pretransformation(..., env, ...)`
already handles `env=None` (fresh Environment with no settings returns `None`
from `_to_lowlevel()`), so all existing tests continue to pass unchanged.

### In `__init__.py`

```python
from .environment import Environment
```

Add `"Environment"` to `__all__`.

### In `cmd/api/main.py`

Uncomment the existing TODO block (lines ~857–878) that creates an
`Environment()` and calls `env.set_docker()` / `env.set_conda_env()`. Update
the import to `from seamless_transformer.environment import Environment`.

---

## Step 3 — Create `seamless_transformer/languages/` module

A new package `seamless_transformer.languages` with a public API for
registering compiled language definitions.

### File structure

```text
seamless_transformer/languages/
    __init__.py          # define_compiled_language(), get_language(), registry
    native/
        __init__.py      # auto-imports all definition files below
        c.py
        cpp.py
        fortran.py
        rust.py
```

### Registry API

```python
# seamless_transformer/languages/__init__.py

_registry: dict[str, LanguageDefinition] = {}

@dataclass
class CompilationConfig:
    compiler: str
    mode: str                    # "object" or "archive"
    options: list[str]
    debug_options: list[str]
    profile_options: list[str]
    release_options: list[str]
    compile_flag: str            # e.g. "-c" for gcc, "" for rustc
    output_flag: str             # e.g. "-o"
    language_flag: str           # e.g. "-x c" for gcc, "" for rustc

@dataclass
class LanguageDefinition:
    name: str
    compilation: CompilationConfig

def define_compiled_language(name: str, compilation: dict) -> None:
    """Register a compiled language definition.

    Example:
        define_compiled_language(
            name="rust",
            compilation={
                "compiler": "rustc",
                "mode": "archive",
                "options": ["--crate-type=staticlib"],
                "debug_options": ["--crate-type=staticlib"],
                "profile_options": [],
                "release_options": [],
                "compile_flag": "",
                "output_flag": "-o",
                "language_flag": "",
            }
        )
    """
    config = CompilationConfig(**compilation)
    if config.mode not in ("object", "archive"):
        raise ValueError(f"Unsupported compile mode: {config.mode!r}")
    _registry[name] = LanguageDefinition(name=name, compilation=config)

def get_language(name: str) -> LanguageDefinition:
    """Look up a registered compiled language.

    Raises KeyError if the language has not been registered.
    """
    if name not in _registry:
        raise KeyError(f"Unknown compiled language: {name!r}")
    return _registry[name]

def is_compiled_language(name: str) -> bool:
    return name in _registry

# Auto-import native definitions
from . import native  # noqa: F401, E402
```

### Native language definitions

Each file in `native/` calls `define_compiled_language()` at import time.

```python
# seamless_transformer/languages/native/c.py
from seamless_transformer.languages import define_compiled_language

define_compiled_language(
    name="c",
    compilation={
        "compiler": "gcc",
        "mode": "object",
        "options": ["-O3", "-ffast-math", "-march=native", "-fPIC", "-fopenmp"],
        "debug_options": ["-g", "-fPIC"],
        "profile_options": [],
        "release_options": [],
        "compile_flag": "-c",
        "output_flag": "-o",
        "language_flag": "-x c",
    }
)
```

```python
# seamless_transformer/languages/native/cpp.py
define_compiled_language(
    name="cpp",
    compilation={
        "compiler": "g++",
        "mode": "object",
        "options": ["-O3", "-ffast-math", "-march=native", "-fPIC", "-fopenmp"],
        "debug_options": ["-g", "-fPIC"],
        "profile_options": [],
        "release_options": [],
        "compile_flag": "-c",
        "output_flag": "-o",
        "language_flag": "-x c++",
    }
)
```

```python
# seamless_transformer/languages/native/fortran.py
define_compiled_language(
    name="fortran",
    compilation={
        "compiler": "gfortran",
        "mode": "object",
        "options": ["-O3", "-fno-automatic", "-fcray-pointer",
                    "-ffast-math", "-march=native", "-fPIC"],
        "debug_options": ["-g", "-fPIC"],
        "profile_options": [],
        "release_options": [],
        "compile_flag": "-c",
        "output_flag": "-o",
        "language_flag": "-x f95",
    }
)
```

```python
# seamless_transformer/languages/native/rust.py
define_compiled_language(
    name="rust",
    compilation={
        "compiler": "rustc",
        "mode": "archive",
        "options": ["--crate-type=staticlib"],
        "debug_options": ["--crate-type=staticlib"],
        "profile_options": [],
        "release_options": [],
        "compile_flag": "",
        "output_flag": "-o",
        "language_flag": "",
    }
)
```

```python
# seamless_transformer/languages/native/__init__.py
from . import c, cpp, fortran, rust  # noqa: F401
```

---

## Step 4 — Create `CompiledTransformer` (mixin + classes)

New module `seamless_transformer/compiled_transformer.py`.

### Architecture

`CompiledTransformer` is implemented as a **mixin class** (`CompiledMixin`)
that is combined with the refactored shared transformer core.

```text
TransformerCore
    ├─ PythonMixin + TransformerCore -> Transformer / DirectTransformer
    └─ CompiledMixin + TransformerCore -> CompiledTransformer / DirectCompiledTransformer
```

### Construction contract

Compiled transformers do **not** call the Python-specific transformer
initialization path. After Step 0, Python callable/text initialization lives in
`PythonMixin`, and compiled transformers build on the shared transformer core
instead.

Instead, `CompiledMixin.__init_compiled__()` must initialize the shared
transformer-core state directly, matching the fields that the runtime relies
on:

```python
self._args = {}
self._modules = {}
self._globals = {}
self._celltypes = {"result": "mixed"}
self._meta = {"transformer_path": ["tf", "tf"], "local": local}
self._scratch = scratch
self._compiled_language = language
self.compilation = deepcopy(lang_def.compilation)
self._environment = Environment()
self._schema_text = None
self._schema = None
self._code_text = None
self._metavars = MetaVars()
self._objects = ObjectList()
```

Required semantics:

- `.language` is read-only and returns `self._compiled_language`.
- `.code` is the compiled source string, not a `Buffer`.
- `.args` remains available for pre-binding inputs.
- `.celltypes` is not configurable for compiled transformers. All Seamless pins
    use `"mixed"`; ABI typing comes from the schema.
- `.modules` and `.globals` are unsupported. If either mapping is non-empty at
    call time, raise `NotImplementedError`.
- `CompiledTransformer.__call__()` returns a delayed `Transformation`.
- `DirectCompiledTransformer.__call__()` delegates to the delayed path and then
    uses the same immediate compute/run sequence as `DirectTransformer`.

### `CompiledMixin` responsibilities

1. **Constructor** takes `language` argument.
   - Calls `get_language(language)` from `seamless_transformer.languages`.
   - Raises `KeyError` if not registered.
   - Test-imports `seamless_signature`: raises `ImportError` with a clear
     message if not installed.
   - Stores `.compilation` (mutable `CompilationConfig` copy).
   - `.language` is read-only after construction (immutable).

2. **`.schema`** attribute (string).
   - Setter also accepts `pathlib.Path` — reads file contents immediately.
   - Parsed by `seamless_signature.load_signature()` (from the YAML string)
     or `Signature.from_dict()` to validate.
   - Stored as the raw YAML string.

3. **`.code`** attribute (string).
   - Setter also accepts `pathlib.Path` — reads file contents immediately.
   - Stored as a raw string (the compiled source code).

4. **`.header`** read-only property.
   - Generated from `.schema` via `seamless_signature.generate_header()`.
   - Returns `None` if schema is not set.

5. **`.metavars`** attribute.
   - If the schema has output-only wildcard dimensions (e.g. `K`), the
     metavars namespace exposes `.maxK` for user assignment.
   - Changing the schema rebuilds metavars, dropping stale entries.
   - A `MetaVars` helper class with `__getattr__` / `__setattr__`.

6. **`.objects`** attribute.
   - A list-like container (`ObjectList`) of `CompiledObject` instances.
   - Supports `append()` and index access.

7. **Callable behavior**.
   - Requires `.schema`, `.code`, and all required metavars to be set.
   - Generates a Python `inspect.Signature` from the seamless-signature
     `Signature` object (mapping input parameters to Python parameters,
     output names to the result).
   - Calling creates a `Transformation` object (delayed) or executes
     immediately (direct), depending on the concrete class.
   - The transformation dict includes `__language__` set to the compiled
     language name, plus `__compiled__: True`, the code, schema, header,
     compilation config, objects, and metavars.

### `CompiledObject`

A simplified version of `CompiledMixin` without the schema:

```python
class CompiledObject:
    def __init__(self, *, language: str):
        # Can be a different language from parent CompiledTransformer
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
        self._code = value
```

Compilation target is `.o` (mode `"object"`) or `.a` (mode `"archive"`),
determined by the object's language's `compilation.mode`.

### `MetaVars` helper

```python
class MetaVars:
    """Dynamic attribute namespace for output-wildcard max-values."""

    def __init__(self):
        self._allowed: set[str] = set()
        self._values: dict[str, int] = {}

    def _rebuild(self, output_wildcards: tuple[str, ...]):
        new_allowed = {f"max{w}" for w in output_wildcards}
        # Remove stale entries
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
```

### Schema Validation Contract

When `.schema` is assigned:

- Accept either a YAML string or a `pathlib.Path`, reading path contents
    immediately.
- Parse into a `seamless_signature.schema.Signature` object and store both the
    raw YAML text and the parsed object.
- If any input or output parameter uses `StructDType`, raise
    `NotImplementedError("Structured dtypes are not supported yet")`.
- Rebuild `self._metavars` from `sig.output_wildcards`.
- Rebuild the Python `inspect.Signature` from `sig.inputs` only.

Output-count semantics:

- One output remains the trivial case.
- Multiple outputs are supported.
- The compiled transformer result celltype must then be either `"mixed"` or
    `"deepcell"`.
- If the user explicitly sets `tf.celltypes.result`, any value other than
    `"mixed"` or `"deepcell"` must raise `TypeError` for multi-output schemas.
- If the user does not set a result celltype explicitly, the default is
    `"mixed"`.

The generated Python callable signature must contain exactly the schema input
names in order, all as positional-or-keyword parameters with no defaults.

### Mixin Sketch

```python
class CompiledMixin:
    """Mixin that adds compiled-language attributes to a Transformer."""

    def __init_compiled__(self, language: str):
        try:
            import seamless_signature  # noqa: F401
        except ImportError:
            raise ImportError(
                "seamless-signature is required for compiled transformers. "
                "Install it with: pip install seamless-signature"
            ) from None
        from seamless_transformer.languages import get_language
        lang_def = get_language(language)
        self._compiled_language = language
        self.compilation = deepcopy(lang_def.compilation)
        self._schema_text = None
        self._schema = None
        self._code_text = None
        self._metavars = MetaVars()
        self._objects = ObjectList()

    # properties: language (read-only), schema, code, header, metavars, objects
    ...


class CompiledTransformer(CompiledMixin, Transformer):
    def __init__(self, language: str, **kwargs):
        self.__init_compiled__(language)
        # Transformer.__init__ with language set, no callable code
        ...

    def __call__(self, *args, **kwargs) -> Transformation:
        # Build pretransformation with compiled info
        ...


class DirectCompiledTransformer(CompiledMixin, DirectTransformer):
    def __init__(self, language: str, **kwargs):
        self.__init_compiled__(language)
        ...

    def __call__(self, *args, **kwargs):
        tf = CompiledTransformer.__call__(self, *args, **kwargs)
        tf._compute(api_origin="call")
        return tf.run()
```

---

## Step 5 — Pretransformation support for compiled transformers

Extend `pretransformation.py` to handle compiled transformation dicts.

### Transformation Payload Contract

Compiled transformations must stay compatible with the current generic
pretransformation and namespace code, which expects every non-dunder entry to
be a standard three-tuple `(celltype, subcelltype, value_or_checksum)`.

Therefore, the compiled payload is:

```python
{
    "__language__": "c",
    "__output__": ("result", "mixed", None),
    "__compiled__": True,
    "__schema__": <checksum hex of schema YAML>,
    "__header__": <checksum hex of generated C header>,
    "__compilation__": <checksum hex of main compilation config>,
    "__env__": <checksum hex of env dict, optional>,
    "__meta__": {
        ...,
        "metavars": {"maxK": 100},
    },
    "code": ("text", "transformer", <compiled source string or checksum>),
    "objects": ("plain", "compiled-objects", [
        {
            "name": "obj0",
            "language": "fortran",
            "code": <source string or checksum>,
            "compilation": <dict>,
        },
    ]),
    "arg1": ("mixed", None, <value_or_checksum>),
}
```

`objects` is a normal pin because it is determinant. It uses celltype
`"plain"` so the existing checksum machinery can serialize it as ordinary data.
Its entries contain object source code plus per-object compilation overrides.

When `CompiledTransformer.__call__` is invoked, it must produce a
pretransformation dict that includes:

```python
{
    # determinant payload
    "__language__": "c",
    "__output__": ("result", "mixed", None),
    "code": ("text", "transformer", <checksum of source code>),
    "objects": ("plain", "compiled-objects", [
        {"name": "obj0", "language": "fortran", "code": <checksum>, ...},
    ]),
    # ordinary input pins...

    # execution-only dunders
    "__compiled__": True,
    "__compilation__": <checksum of compilation config dict>,
    "__schema__": <checksum of schema YAML>,
    "__header__": <checksum of generated C header>,
    "__meta__": { ..., "metavars": {"maxK": 100} },
    "__env__": <checksum of env dict>,
}
```

**Naming convention**: dunder keys (`__x__`) are reserved for attributes that
do not affect the transformation result (or are grandfathered in, like
`__language__` and `__output__` — see
`legacy-seamless/seamless/util/transformation.py` line 17). Determinant keys
— those that change the result when they change — use plain names. For
compiled transformers, `code` and `objects` are determinant (different source
code produces different results), so they must be plain keys. This matches
how bash transformations already use `code` as a plain key rather than
`__code__`.

**Compilation config is execution-only, not determinant.** Compiler flags
(`-O3`, `-ffast-math`, `-march=native`, debug vs release) are analogous to
the Python interpreter version or BLAS backend — they are nuisance
parameters under the env-null-hypothesis. The scientifically meaningful
result should be invariant under compiler flag variation. If it isn't, that
is detected through recomputation and witness comparison, not by baking
flags into the cache key. This matches legacy Seamless, where `__compilers__`
and `__languages__` were dunders excluded from the transformation checksum.
Compilation config for objects is likewise part of their `__compilation__`,
not stored inside the determinant `objects` list.

### Required Checksum-Path Changes

To make the above rule real, the implementation must update the checksum and
transport helpers rather than merely documenting the intent.

In `transformation_utils.py`:

- Introduce `TRANSFORMATION_EXECUTION_DUNDER_KEYS = {"__compiled__",
    "__compilation__", "__schema__", "__header__"}`.
- Make `tf_get_buffer()` exclude those keys from the checksum-defining buffer.
- Make `extract_tf_dunder()` and `extract_job_dunder()` include those keys so
    they still travel to workers/jobserver execution.

In `pretransformation.py`:

- Add the same keys to `NON_CHECKSUM_ITEMS`.

In `transformation_namespace.py` and `run.py`:

- Skip those dunders in generic pin iteration.
- Resolve their referenced buffers explicitly when `__compiled__` is true.

This is not optional. Without these edits, the current code would still hash
the new compiled dunders into the transformation checksum.

A new function `compiled_transformer_to_pretransformation()` in
`pretransformation.py` handles this. It reuses the existing machinery for
checksumming arguments and building the pretransformation dict, adding the
compiled-specific keys.

The Python signature for callable behavior is generated from the schema:

- Each schema input becomes a Python parameter.
- Schema outputs map to a single Seamless result pin named `"result"`.
- Wildcards used only in outputs require corresponding `metavars.maxX` to be
  set.

### Result Contract

Phase 1 supports one or more schema outputs.

- For a single output:
    The compiled transformer may return the bare scalar/array value, matching the
    current single-output expectation.
- For multiple outputs with result celltype `"mixed"`:
    The compiled transformer returns a Python dict keyed by schema output name.
    Each dict value is either a Python scalar or a numpy array.
- For multiple outputs with result celltype `"deepcell"`:
    The compiled transformer returns a Python dict keyed by schema output name,
    but before the final result buffer is created each leaf value is packed as an
    individual checksum-addressed element so downstream Seamless code sees a true
    deep structure.
- Supporting `"deepcell"` removes the earlier concern that multi-output values
    would only be cached as one undifferentiated mixed object.
- The implementation must verify both multi-output result modes with tests.

---

## Step 6 — Build pipeline: compilation + CFFI + `.so`

This is the runtime execution path for compiled transformers. It ports and
modernizes the legacy `build_compiled_module` → `compile` → `cffi` →
`build_extension_cffi` → `import_extension_module` pipeline.

### New modules

```text
seamless_transformer/
    compiler/
        __init__.py          # public API: build_compiled_module()
        compile.py           # compile(): source → binary objects (.o / .a)
        cffi_wrapper.py      # CFFI wrapper generation and extension building
        locks.py             # file-based directory lock (ported from legacy)
```

### 6a — `compiler/compile.py`: source → binary objects

Ported from legacy `seamless/compiler/compile.py`. Takes a completed module
definition (with resolved compiler paths, flags, etc.) and runs the compiler
to produce binary objects.

**Changes from legacy:**

- Uses `subprocess.run()` with explicit arguments list (no `shell=True`) for
  security. The command is constructed as a list, not a string.
- Removes Silk dependency (`complete()` takes plain dicts, not Silk objects).
- The `complete()` function resolves compilation config from `CompilationConfig`
  objects (from the language registry) rather than from cson tables.
- Source files are written with a generic name (`code`) — the compiler is
  told which language to use via `language_flag` (e.g. `-x c` for gcc,
  `-x c++` for g++, `-x f95` for gfortran). Compilers that don't need it
  (rustc) have an empty `language_flag`. This eliminates the need for a
  file extension mapping.
- Build dir is a `tempfile.TemporaryDirectory` context manager for automatic
  cleanup.

**Pipeline:**

```text
For each object in module_definition["objects"]:
    1. Write header files to build dir
    2. Write source code to build dir as "code" (no extension)
    3. Run compiler with language_flag:
       - mode "object":  gcc -x c -c -O3 ... -o objname.o code
       - mode "archive": rustc --crate-type=staticlib -o objname.a code
    4. Read binary output into dict
```

**`_merge_objects()`**: Same logic as legacy — handles `"object"` mode
(one `.o` per source) and `"archive"` mode (one `.a`). `"package"` mode is
not implemented.

**`complete()`**: Takes the raw module definition from `CompiledTransformer`
and fills in compiler binary path, flags for the chosen target
(debug/profile/release), etc., using the `CompilationConfig` from the
language registry.

```python
def complete(module_definition: dict, languages_registry) -> dict:
    """Complete a module definition using the language registry."""
    m = deepcopy(module_definition)
    target = m.get("target", "profile")
    m["target"] = target
    m["link_options"] = m.get("link_options", [])

    for objname, obj in m["objects"].items():
        lang_name = obj["language"]
        lang_def = languages_registry.get_language(lang_name)
        comp = lang_def.compilation

        obj["compile_mode"] = comp.mode
        obj["compiler_binary"] = comp.compiler  # or shutil.which(comp.compiler)
        obj["compile_flag"] = comp.compile_flag
        obj["output_flag"] = comp.output_flag
        obj["language_flag"] = comp.language_flag

        # Resolve options for target
        if target in ("release", "profile"):
            options = list(obj.get("options", comp.options))
            if target == "profile":
                options += list(comp.profile_options)
        else:
            options = list(obj.get("debug_options", comp.debug_options))
        obj["options"] = options

    return m
```

### 6b — `compiler/cffi_wrapper.py`: CFFI wrapper + extension building

Ported from legacy `seamless/compiler/cffi.py` and
`seamless/compiler/build_extension.py`.

**What it does:**

1. Takes the C header (generated by seamless-signature) and generates CFFI
   wrapper C source code using `cffi.FFI` + `cffi.recompiler.Recompiler`.
2. Links the CFFI wrapper C source with the compiled binary objects (`.o` /
   `.a` files) to produce a Python extension `.so`.
3. Returns the `.so` bytes.

**Changes from legacy:**

- Uses `setuptools` instead of `distutils` (which is removed in Python 3.12).
  Specifically: `setuptools.Extension` and `setuptools.Distribution`.
- Same temp-directory lifecycle: create temp dir, write objects and CFFI
  wrapper, build, read `.so`, clean up.
- Explicit `cffi` availability check with a clear error message.

```python
def build_extension_cffi(
    full_module_name: str,
    binary_objects: dict[str, bytes],
    target: str,
    c_header: str,
    link_options: list[str],
    compiler_verbose: bool = False,
) -> bytes:
    """Build a Python extension .so from binary objects and a C header.

    Uses CFFI to generate a wrapper that exposes the C functions declared
    in c_header, links them with the binary objects, and returns the .so
    file contents as bytes.
    """
    try:
        from cffi import FFI
        from cffi.recompiler import Recompiler
    except ImportError:
        raise ImportError(
            "cffi is required for compiled transformers. "
            "Install it with: pip install cffi"
        ) from None

    from setuptools import Extension, Distribution
    ...
```

### 6c — `compiler/__init__.py`: orchestrator

Ported from legacy `build_compiled_module()` in `build_module.py`.

**What it does:**

1. Takes a module definition (from `CompiledTransformer`) with `objects`,
   `public_header`, `link_options`, `target`.
2. Calls `complete()` to resolve compiler paths and flags.
3. For each object: check cache → if miss, call `compile()`.
4. Call `build_extension_cffi()` to link everything into a `.so`.
5. Import the `.so` as a Python module via `importlib`.
6. Return the module (with `.lib` attribute exposing C functions).

**Caching:**

- Object-level caching: binary objects (`.o` / `.a`) cached by checksum of
  their source + compiler config.
- Module-level caching: final `.so` cached by checksum of the complete
  module definition.
- Uses `functools.lru_cache` or a simple dict cache.
- Database/remote caching hooks are left as stubs (matching legacy's
  disabled code).

```python
def build_compiled_module(
    module_definition: dict,
    *,
    module_name: str | None = None,
) -> ModuleType:
    """Build and import a compiled Python extension module.

    Pipeline:
        module_definition → complete() → compile() → build_extension_cffi()
        → import_extension_module() → Python module with .lib attribute
    """
    ...
```

### 6d — `compiler/locks.py`: directory lock

Ported directly from legacy `seamless/compiler/locks.py`. A file-based mutex
for thread-safe compilation to shared temp directories. No changes needed.

### 6e — Integration with `run.py`

When `run.py` encounters a transformation with `__compiled__: True`:

1. Resolve schema, code, objects, compilation config from checksums.
2. Build the module definition dict:

   ```python
   module_definition = {
       "type": "compiled",
       "objects": {
           "main": {"code": code, "language": language},
           **additional_objects
       },
       "public_header": {"language": "c", "code": c_header},
       "link_options": link_options,
       "target": target,
   }
   ```

3. Call `build_compiled_module(module_definition)`.
4. Call the `transform()` function on the result module's `.lib` attribute,
   passing the input values with appropriate type conversions.

**Type marshalling** (inputs → C types → outputs):

The C header generated by seamless-signature defines the function signature.
CFFI handles the Python↔C type conversion, but the caller must build the exact
argument list in the same order as `seamless_signature.generate_header()`:

1. Input wildcard values in `sig.input_wildcards` order.
2. Output wildcard maxima in `sig.output_wildcards` order.
3. Input parameters in `sig.inputs` order.
4. Output wildcard result pointers in `sig.output_wildcards` order.
5. Output parameter buffers in `sig.outputs` order.

`call_compiled_transform()` implements this contract exactly.

### ABI Marshalling Rules

#### Input wildcards

- For every wildcard in `sig.input_wildcards`, derive the concrete runtime size
    from the provided numpy input arrays.
- All parameters that reference the same wildcard must agree on the resolved
    size, or raise `ValueError` before calling into C.
- Pass these sizes as Python integers; CFFI converts them to `unsigned int`.

#### Output-only wildcards

- For every wildcard in `sig.output_wildcards`, require the corresponding
    `tf.metavars.maxX` value to be present.
- Pass `maxX` as the `unsigned int maxX` argument.
- Also allocate `ffi.new("unsigned int *")` for the runtime-returned `X`
    argument and pass that pointer.
- After the C call returns, read back the actual output size from the pointer.

#### Scalar parameters

- Supported scalar dtypes are those that `seamless-signature` maps to C scalar
    types.
- For scalar inputs, accept either plain Python scalar values (`int`, `float`,
    `bool`) or native-endian numpy scalar values whose dtype matches the schema.
- For plain Python scalar inputs, pass the value through directly and rely on
    CFFI to coerce it to the target C scalar type.
- For numpy scalar inputs, require native endianness and exact dtype match with
    the schema, then pass the scalar value through directly and rely on CFFI to
    coerce it to the target C scalar type.
- The implementation must not silently reinterpret non-native-endian numpy
    scalar values.
- For scalar outputs, allocate `ffi.new("<ctype> *")` and dereference after the
    call.

#### Array parameters

- Array inputs and outputs require `numpy`.
- Accept only native-endian numpy arrays with dtype exactly matching the
    schema's scalar dtype.
- Input arrays must be normalized to satisfy all of: native-endian,
    C-contiguous, and aligned. The plan-level intent is equivalent to using
    `np.require(value, dtype=expected_dtype, requirements=["C", "ALIGNED"])`,
    with the additional rule that byte order must already be native-endian or a
    safe explicit conversion must be performed before the CFFI call.
- Output arrays must be allocated as native-endian, C-contiguous, aligned numpy
    arrays. The plan-level intent is equivalent to `np.empty(shape,
    dtype=expected_dtype, order="C")`; the implementation must verify that the
    resulting array is aligned before handing its buffer to CFFI.
- Compute the full shape as `resolved_wildcard_prefix + parameter.element_shape`.
- For output arrays with output-only wildcards, allocate using the wildcard
    maxima, call into C, then slice down to the returned runtime shape.
- Pass array buffers using `ffi.from_buffer()`.

### Required CFFI Verification

The implementation must verify, with tests, that the assumed CFFI coercions and
buffer handling actually work in the supported cases. This is a hard
requirement, not an implementation note.

At minimum, the agent must add tests that demonstrate all of the following:

- A compiled transformer accepts plain Python scalar inputs for each scalar type
    used in the tests and produces the correct result.
- A compiled transformer accepts native-endian numpy scalar inputs and produces
    the correct result.
- A compiled transformer accepts aligned, C-contiguous native-endian numpy
    array inputs and produces the correct result.
- Output arrays allocated by the implementation are accepted by CFFI and yield
    correct values after slicing to runtime shape.
- If the implementation chooses to normalize non-contiguous input arrays, that
    normalization path is exercised by a test.
- Non-native-endian numpy scalars or arrays are either rejected explicitly or
    converted explicitly, and the chosen behavior is covered by tests.

#### Unsupported schema features in phase 1

- `StructDType` parameters are rejected before execution.

### Required Module-Definition Shape

`run.py` must construct the compiler input with this exact schema:

```python
module_definition = {
        "type": "compiled",
        "target": target,
        "link_options": [],
        "public_header": {"language": "c", "code": c_header},
        "objects": {
                "main": {
                        "language": language,
                        "code": code,
                        "options": compilation_override.get("options"),
                        "debug_options": compilation_override.get("debug_options"),
                },
                "obj0": {
                        "language": "fortran",
                        "code": object_code,
                        "options": object_override.get("options"),
                },
        },
}
```

`build_compiled_module()` and `complete()` must consume this exact object-map
format, not a list.

This marshalling logic lives in a new helper function
`call_compiled_transform()` that sits alongside `build_compiled_module()`.

---

## Step 7 — Tests

### 7a — Environment tests (`tests/test_environment.py`)

Unit tests for `Environment` (no Seamless infrastructure needed):

- `_save()` returns `None` for a fresh instance.
- `_save()` / `_load()` round-trip preserves all props.
- `_to_lowlevel()` returns `None` for a fresh instance.
- `_to_lowlevel()` emits `"conda_environment"` after `set_conda_env()`.
- `set_which()` validates input types.
- `set_docker()` validates presence of `"name"` key.
- `set_conda()` validates YAML string is a dict with `"dependencies"`.

Regression: all existing tests pass unchanged (fresh `Environment()` returns
`None` from `_to_lowlevel()`, matching the previous `env = None`).

### 7b — Language registry tests (`tests/test_languages.py`)

- Native languages (c, cpp, fortran, rust) are registered on import.
- `get_language("c")` returns the correct `LanguageDefinition`.
- `is_compiled_language("python")` returns `False`.
- `define_compiled_language()` with invalid mode raises `ValueError`.
- Custom language registration works and is retrievable.

### 7c — CompiledTransformer API tests (`tests/test_compiled_transformer.py`)

- Construction with valid language succeeds.
- Construction with unregistered language raises `KeyError`.
- Construction without seamless-signature installed raises `ImportError`
  (mock the import).
- `.language` is immutable after construction.
- `.schema` setter accepts `str` and `pathlib.Path`.
- `.code` setter accepts `str` and `pathlib.Path`.
- `.header` generates correct C header from schema.
- `.metavars` exposes `maxK` when schema has output wildcard `K`.
- `.metavars` drops stale entries when schema changes.
- Setting a schema with structured dtypes raises `NotImplementedError`.
- A multi-output schema is accepted.
- For a multi-output schema, result celltype `"mixed"` is accepted.
- For a multi-output schema, result celltype `"deepcell"` is accepted.
- For a multi-output schema, any other result celltype is rejected.
- Not callable when schema/code/metavars incomplete.
- Callable when all are set; produces a `Transformation` object.
- `.objects` supports `append()` and index access.
- `CompiledObject` has its own language (can differ from parent).

### 7d — Compilation pipeline tests (`tests/test_compiler.py`)

These tests require a C compiler (gcc) to be installed. They can be marked
with `@pytest.mark.skipif(not shutil.which("gcc"))`.

- `complete()` resolves compiler config from language registry.
- `compile()` compiles a simple C source to a `.o` file.
- `_merge_objects()` handles `"object"` and `"archive"` modes.
- `build_extension_cffi()` links a `.o` with a CFFI wrapper to produce a
  `.so` (requires cffi).
- `build_compiled_module()` full pipeline: C source → `.so` → importable
  module with correct `transform()` result.
- A focused CFFI coercion test verifies that plain Python scalars and
    native-endian numpy scalars are both accepted for scalar arguments.
- A focused buffer test verifies that aligned native-endian numpy arrays are
    accepted for array arguments.
- A focused result-packing test verifies that multi-output results can be
    packaged as `"mixed"` and as `"deepcell"`.

### 7e — End-to-end compiled transformer tests (`tests/test_compiled_e2e.py`)

Full integration tests (require gcc + cffi):

**C test:**

```python
tf = DirectCompiledTransformer("c")
tf.schema = "..."  # simple add(a: int32, b: int32) -> result: int32
tf.code = """
int transform(int32_t a, int32_t b, int32_t *result) {
    *result = a + b;
    return 0;
}
"""
assert tf(a=2, b=3) == 5
```

This test must be exercised twice:

- once with plain Python scalar inputs;
- once with native-endian numpy scalar inputs.

**C++ test (with extern "C"):**

```python
tf = DirectCompiledTransformer("cpp")
tf.schema = ...
tf.code = """
extern "C" int transform(int32_t a, int32_t b, int32_t *result) {
    *result = a + b;
    return 0;
}
"""
```

**Rust test (with archive mode):**

```python
tf = DirectCompiledTransformer("rust")
tf.schema = ...
tf.code = """
#[no_mangle]
pub unsafe extern "C" fn transform(a: i32, b: i32, c: *mut i32) -> i32 {
    (*c) = a + b;
    return 0;
}
"""
```

**Multi-object test (C main + Fortran helper):**

```python
tf = DirectCompiledTransformer("c")
tf.schema = ...
tf.code = main_c_code
obj = CompiledObject(language="fortran")
obj.code = fortran_helper_code
tf.objects.append(obj)
```

**Delayed (non-direct) test:**

```python
tf = CompiledTransformer("c")
tf.schema = ...
tf.code = ...
result = tf(a=2, b=3)  # returns Transformation
assert isinstance(result, Transformation)
value = result.run()    # actually runs
assert value == 5
```

**Multi-output packaging tests:**

```python
tf = DirectCompiledTransformer("c")
tf.schema = "..."  # two outputs
tf.celltypes.result = "mixed"
result = tf(...)
assert isinstance(result, dict)
assert set(result) == {"first", "second"}
```

```python
tf = DirectCompiledTransformer("c")
tf.schema = "..."  # two outputs
tf.celltypes.result = "deepcell"
result = tf(...)
assert isinstance(result, dict)
assert set(result) == {"first", "second"}
```

The implementation must verify not only the returned Python values but also
that the `"deepcell"` mode produces an actual packed deep result, not a plain
mixed dict serialized wholesale.

**Phase-1 negative tests:**

- A schema with a structured dtype is rejected at assignment time.
- If non-native-endian numpy inputs are not normalized, they are rejected with
    an explicit error that is covered by tests.

---

## Dependency changes

### `pyproject.toml` updates

```toml
[project.optional-dependencies]
compiled = ["cffi", "numpy", "seamless-signature"]
```

`cffi`, `numpy`, and `seamless-signature` are optional dependencies — only
needed when using compiled transformers. The code does runtime import checks
with clear error messages.

---

## File summary

New files to create:

| File | Purpose |
| --- | --- |
| `seamless_transformer/environment.py` | `Environment` class |
| `seamless_transformer/languages/__init__.py` | Language registry API |
| `seamless_transformer/languages/native/__init__.py` | Auto-import native defs |
| `seamless_transformer/languages/native/c.py` | C language definition |
| `seamless_transformer/languages/native/cpp.py` | C++ language definition |
| `seamless_transformer/languages/native/fortran.py` | Fortran language definition |
| `seamless_transformer/languages/native/rust.py` | Rust language definition |
| `seamless_transformer/compiled_transformer.py` | `CompiledMixin`, `CompiledTransformer`, `DirectCompiledTransformer`, `CompiledObject`, `MetaVars`, `ObjectList` |
| `seamless_transformer/compiler/__init__.py` | `build_compiled_module()` orchestrator |
| `seamless_transformer/compiler/compile.py` | `compile()`, `complete()` |
| `seamless_transformer/compiler/cffi_wrapper.py` | CFFI wrapper generation + extension building |
| `seamless_transformer/compiler/locks.py` | File-based directory lock |
| `tests/test_environment.py` | Environment unit tests |
| `tests/test_languages.py` | Language registry tests |
| `tests/test_compiled_transformer.py` | CompiledTransformer API tests |
| `tests/test_compiler.py` | Compilation pipeline tests |
| `tests/test_compiled_e2e.py` | End-to-end integration tests |

Files to modify:

| File | Change |
| --- | --- |
| `seamless_transformer/transformer_class.py` | Extract `TransformerCore` and `PythonMixin`; preserve existing Python/bash behavior |
| `seamless_transformer/transformer_class.py` | Wire Environment into Transformer |
| `seamless_transformer/__init__.py` | Export Environment, CompiledTransformer |
| `seamless_transformer/pretransformation.py` | Add compiled pretransformation support |
| `seamless_transformer/run.py` | Add compiled execution path |
| `pyproject.toml` | Add optional `compiled` dependencies |

---

## Implementation order

The steps have dependencies:

```text
Step 0 (TransformerCore + PythonMixin)
    ↓
Step 1 (Environment)
    ↓
Step 2 (Wire into Transformer)     Step 3 (Language registry)
    ↓                                  ↓
    ↓                              Step 4 (CompiledTransformer API)
    ↓                                  ↓
    ↓                              Step 5 (Pretransformation support)
    ↓                                  ↓
    └──────────────────────────→  Step 6 (Build pipeline)
                                       ↓
                                  Step 7 (Tests — can be written alongside each step)
```

Step 0 is mandatory before the compiled transformer implementation.
Steps 1–2 and Step 3 are independent after Step 0 and can be done in parallel.
Steps 4–5 depend on Step 0 and Step 3.
Step 6 depends on Steps 0–2 (for core architecture and env) and Steps 4–5 (for the data model).
Tests are written alongside each step but the integration tests (7d, 7e) need
everything.
