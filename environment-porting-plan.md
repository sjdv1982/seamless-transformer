# Porting Plan: Environment (clearing the ground for compiled language support)

## Goal

Port `Environment` from legacy Seamless into `seamless-transformer`, enabling:

1. `Transformer.environment` to work as a real attribute (not a stub).
2. The `__env__` pipeline that already exists in the current code to be
   populated from user-supplied values.
3. A `ContextEnvironment` stub holding language/compiler tables — the exact
   hook that compiled language support will need in the follow-on step.

The compiled language machinery itself (`htf["compiled"]=True`, header
generation, `build_compiled_module`, CFFI) is **out of scope** here.

---

## Background: what legacy `Environment` does

Legacy Seamless has two classes in
[seamless/Environment.py](legacy-seamless/seamless/Environment.py):

**`Environment`** — per-transformer environment.  
Serializable props (`_props`): `conda`, `conda_env_name`, `which`, `powers`,
`docker`.  
Lives on `Transformer.environment`. Serializes to a state dict via `_save()`,
round-trips with `_load()`, and is embedded in the transformation dict as
`__env__` (a checksum of the state dict serialized as `"plain"`).

**`ContextEnvironment(Environment)`** — global environment for a whole context.  
Adds: `_languages` / `_compilers` (override the shipped `languages.cson` /
`compilers.cson` tables — the compiled-language extensibility hook),
`_ipy_templates`, `_py_bridges`.

The `_to_lowlevel()` method on `Environment` converts the internal state into
the `env_dict` that is checksum-stored as `__env__`:

| `env_dict` key          | source prop          | used by                        |
|-------------------------|----------------------|--------------------------------|
| `conda`                 | `_conda` (YAML dict) | remote/jobserver env setup     |
| `conda_env_name`        | `_conda_env_name`    | Python transformer conda env   |
| `conda_bash_env_name`   | `_conda_env_name`    | bash transformer conda env     |
| `which`                 | `_which`             | pre-flight binary check        |
| `powers`                | `_powers`            | abstract capability dispatch   |
| `docker`                | `_docker`            | Docker/Singularity dispatch    |

---

## Current state in `seamless-transformer`

The `__env__` plumbing already exists end-to-end:

- **[pretransformation.py:268-271](seamless-transformer/seamless_transformer/pretransformation.py#L268-L271)**  
  `direct_transformer_to_pretransformation()` accepts `env`, serializes it as
  `Buffer(env, "plain")`, stores the checksum as `__env__`.

- **[transformation_utils.py:102-110](seamless-transformer/seamless_transformer/transformation_utils.py#L102-L110)**  
  `resolve_env_checksum()` extracts `__env__` from the transformation dict or
  `tf_dunder`.

- **[run.py:122-129](seamless-transformer/seamless_transformer/run.py#L122-L129)**  
  Resolves `__env__` → `env_dict`, passes `env_dict.get("conda_environment", "")`
  into the bash path.

- **[execute_bash.py:119-136](seamless-transformer/seamless_transformer/execute_bash.py#L119-L136)**  
  Uses `conda_environment_` to emit `conda activate` in `transform.sh`.

What is **stubbed out** in
[transformer_class.py](seamless-transformer/seamless_transformer/transformer_class.py):

```python
# __init__, lines 147-149
"""
STUB
self._environment = Environment(self)
self._environment_state = None
"""

# environment property, lines 218-223
'''
STUB
@property
def environment(self) -> "Environment":
    """Computing environment to execute transformations in"""
    return self._environment
/STUB
'''

# __call__, line 253
env = None  # environment handling not ported
```

### Gap table

| Legacy component                              | Current status        | Gap                                        |
|-----------------------------------------------|-----------------------|--------------------------------------------|
| `Environment` class with all `_props`         | Does not exist        | Entire class missing                       |
| `_save()` / `_load()` / `_to_lowlevel()`     | Nothing               | Serialization lifecycle missing            |
| `Transformer._environment` attribute          | Stubbed               | Property/init stubs present, inactive      |
| `env = None` in `Transformer.__call__`        | Hardcoded `None`      | Should call `self._environment._to_lowlevel()` |
| `__env__` → `conda_bash_env_name` (bash)      | **Working**           | `run.py` already picks up `conda_environment` |
| `__env__` → `conda` (full YAML spec)          | Not consumed          | Only named-env path is consumed            |
| `__env__` → `which`                           | Not consumed          | No pre-flight check                        |
| `__env__` → `powers`, `docker`                | Not consumed          | Not interpreted                            |
| `ContextEnvironment` (languages/compilers)    | Does not exist        | Required for compiled language support     |
| `set_ipy_templates`, `set_py_bridges`         | Not present           | Out of scope (separate concern)            |

---

## Step-by-step porting plan

### Step 1 — Create `seamless_transformer/environment.py`

Port `Environment` as a standalone, self-contained class.

**Changes from legacy:**

- Remove `weakref` parent reference and the reactive `_update()` / `_sync()`
  lifecycle methods. In the current design the transformer is functional (not
  reactive): environment state is read once at call time via `_to_lowlevel()`,
  not live-synced to a running graph.
- Keep `_save()` and `_load()` for serialization round-tripping and for use
  by `DirectTransformer._environment_state` if needed later.
- Keep `_to_lowlevel(*, bash=False)` unchanged — this is the bridge to
  `__env__`.
- Keep all public setters/getters: `set_conda`, `set_conda_env`, `get_conda`,
  `set_which`, `get_which`, `set_powers`, `get_powers`, `set_docker`,
  `get_docker`.
- Dependencies: `ruamel.yaml`, `json`, `copy.deepcopy` — same as legacy,
  nothing new.

```python
# seamless_transformer/environment.py

import json
from copy import deepcopy
import ruamel.yaml

yaml = ruamel.yaml.YAML(typ="safe")


class Environment:
    """Execution environment for an individual transformer."""

    _props = ["_conda", "_conda_env_name", "_which", "_powers", "_docker"]

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
        if not state:
            return None
        json.dumps(state)   # validate serializability
        return state

    def _load(self, state: dict):
        for prop in self._props:
            v = state.get(prop[1:])
            if v is not None:
                setattr(self, prop, v)

    def _to_lowlevel(self, *, bash=False) -> dict | None:
        result = {}
        if self._which is not None:
            result["which"] = deepcopy(self._which)
        if self._conda is not None:
            conda_env = yaml.load(self._conda)
            result["conda"] = conda_env
        if self._conda_env_name is not None:
            if bash:
                result["conda_bash_env_name"] = self._conda_env_name
            else:
                result["conda_env_name"] = self._conda_env_name
        if self._powers is not None:
            result["powers"] = deepcopy(self._powers)
        if self._docker is not None:
            result["docker"] = self._docker
        if not result:
            return None
        return result

    # --- public API -----------------------------------------------------------

    def set_conda(self, conda, format="yaml"):
        """Set the conda environment definition (YAML string or file-like)."""
        if format != "yaml":
            raise NotImplementedError(format)
        if conda is None:
            self._conda = None
            return
        if hasattr(conda, "read") and callable(conda.read):
            conda = conda.read()
        elif isinstance(conda, str) and (
            conda.endswith(".yaml") or conda.endswith(".yml")
        ):
            conda = open(conda).read()
        result = yaml.load(conda)
        if not isinstance(result, dict):
            raise TypeError(f"Must be dict, not {type(result)}")
        _ = result["dependencies"]
        self._conda = conda

    def set_conda_env(self, conda_env_name):
        """Name of an existing conda environment to run the transformer in."""
        self._conda_env_name = conda_env_name

    def get_conda(self, format="yaml"):
        if format != "yaml":
            raise NotImplementedError(format)
        return deepcopy(self._conda)

    def set_which(self, which, format="plain"):
        """List of binaries that must be present on PATH."""
        if format != "plain":
            raise NotImplementedError(format)
        if which is None:
            self._which = None
            return
        if not isinstance(which, list):
            raise TypeError(f"Must be list, not {type(which)}")
        for w in which:
            if not isinstance(w, str):
                raise TypeError("Must be list of strings")
        self._which = which

    def get_which(self, format="plain"):
        if format != "plain":
            raise NotImplementedError(format)
        return deepcopy(self._which)

    def set_powers(self, powers):
        """List of abstract capabilities required (e.g. 'docker', 'ipython')."""
        if powers is None:
            self._powers = None
            return
        if not isinstance(powers, list):
            raise TypeError(f"Must be list, not {type(powers)}")
        for p in powers:
            if not isinstance(p, str):
                raise TypeError("Must be list of strings")
        self._powers = powers

    def get_powers(self):
        return deepcopy(self._powers)

    def set_docker(self, docker: dict):
        """Docker config dict; must contain at least 'name'."""
        if docker is not None:
            if not isinstance(docker, dict):
                raise TypeError(f"Must be dict, not {type(docker)}")
            if "name" not in docker:
                raise ValueError('Docker dict must contain at least "name"')
        self._docker = deepcopy(docker)

    def get_docker(self):
        return deepcopy(self._docker)


class ContextEnvironment(Environment):
    """Environment stub for a whole context — holds language/compiler tables.

    Only the language and compiler table API is ported here; ipy_templates and
    py_bridges are left for a later step.
    """

    _props = Environment._props + ["_languages", "_compilers"]

    def __init__(self):
        super().__init__()
        self._languages = None
        self._compilers = None

    def set_languages(self, languages, format="plain"):
        """Override the shipped language table.

        format='plain'  — pass a dict directly.
        format='cson'   — pass a CSON string (requires seamless.util.cson2json).
        Setting languages=None resets to the shipped default.
        """
        if format not in ("cson", "plain"):
            raise NotImplementedError(format)
        if languages is None:
            self._languages = None
            return
        if format == "cson":
            from seamless.util import cson2json
            result = cson2json(languages)
            self._languages = languages  # stored as cson, like legacy
        else:
            json.dumps(languages)        # validate serializability
            self._languages = json.dumps(languages, sort_keys=True, indent=2)

    def get_languages(self, format="plain"):
        """Return the current language table (falls back to shipped default)."""
        from seamless.util import cson2json

        if format not in ("cson", "plain"):
            raise NotImplementedError(format)
        if self._languages is None:
            from seamless.compiler import languages_cson as default
            src = default
        else:
            src = self._languages
        if format == "cson":
            return deepcopy(src)
        return cson2json(src)

    def set_compilers(self, compilers, format="plain"):
        """Override the shipped compiler table.

        format='plain'  — pass a dict directly.
        format='cson'   — pass a CSON string.
        Setting compilers=None resets to the shipped default.
        """
        if format not in ("cson", "plain"):
            raise NotImplementedError(format)
        if compilers is None:
            self._compilers = None
            return
        if format == "cson":
            from seamless.util import cson2json
            result = cson2json(compilers)
            self._compilers = compilers  # stored as cson
        else:
            json.dumps(compilers)        # validate serializability
            self._compilers = json.dumps(compilers, sort_keys=True, indent=2)

    def get_compilers(self, format="plain"):
        """Return the current compiler table (falls back to shipped default)."""
        from seamless.util import cson2json

        if format not in ("cson", "plain"):
            raise NotImplementedError(format)
        if self._compilers is None:
            from seamless.compiler import compilers_cson as default
            src = default
        else:
            src = self._compilers
        if format == "cson":
            return deepcopy(src)
        return cson2json(src)

    def _find_language(self, language):
        """Resolve a language name against the active language table."""
        from seamless.compiler import find_language
        return find_language(language, self.get_languages("plain"))
```

---

### Step 2 — Wire `environment` into `Transformer`

In [transformer_class.py](seamless-transformer/seamless_transformer/transformer_class.py):

**In `__init__`**, replace the two-line STUB:

```python
# Remove:
"""
STUB
self._environment = Environment(self)
self._environment_state = None
"""

# Add (at the same indentation level, after self._set_code):
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
env = self._environment._to_lowlevel(bash=(self._language == "bash"))
```

`DirectTransformer` inherits all of this automatically.

---

### Step 3 — Export from the package

In [seamless_transformer/**init**.py](seamless-transformer/seamless_transformer/__init__.py):

```python
from .environment import Environment, ContextEnvironment

__all__ = [
    ...
    "Environment",
    "ContextEnvironment",
]
```

---

### Step 4 — Tests

**Unit tests** for `Environment` (no Seamless infrastructure needed):

- `_save()` returns `None` for a fresh instance.
- `_save()` / `_load()` round-trip preserves all props.
- `_to_lowlevel(bash=False)` maps `conda_env_name` → `conda_env_name`.
- `_to_lowlevel(bash=True)` maps `conda_env_name` → `conda_bash_env_name`.
- `_to_lowlevel()` returns `None` for a fresh instance.

**Integration test** for the bash path (the one working path today):

```python
@direct
def mytf():
    result = "ok"

mytf.language = "bash"
mytf.environment.set_conda_env("myenv")
# call and assert transform.sh contains "conda activate myenv"
```

**Regression**: run the full existing test suite — all tests currently pass
`env=None`, so `_to_lowlevel()` returning `None` for a fresh `Environment()`
must leave the existing behaviour unchanged.

---

### Step 5 — Note on `_to_lowlevel` key naming

Legacy uses two different key names depending on `bash=`:

- `bash=False` → `"conda_env_name"`
- `bash=True`  → `"conda_bash_env_name"`

But `run.py` line 189 currently reads:

```python
env_dict.get("conda_environment", "")
```

This key name (`conda_environment`) matches neither legacy key. This is a
pre-existing inconsistency in the current code — the bash path happens to
work only because `execute_bash.py` is invoked with this key name.

**Action**: when wiring up `_to_lowlevel`, verify whether `run.py` should read
`conda_bash_env_name` (matching legacy) or whether `_to_lowlevel` should emit
`conda_environment` (matching the current consumer). Align the two; do not
silently leave them mismatched.

---

## Out of scope (follow-on: compiled language support)

- `Transformer.language` routing into a compiled path (`htf["compiled"]=True`,
  schema-driven C-header generation, `build_compiled_module`, CFFI).
- `ContextEnvironment` wired to a `Context` class (no `Context` exists yet
  in seamless-transformer).
- `_parse_and_validate` (pre-flight `which` check).
- `_ipy_templates`, `_py_bridges`.

The `ContextEnvironment` stub created in Step 1 is the landing zone for the
compiled language work: once it is present, language and compiler tables can
be set, and the compiled transformer translator can call `_find_language()`.

## Note

One thing worth flagging that emerged during the analysis — see Step 5 in the plan. There's a pre-existing key name mismatch: _to_lowlevel(bash=True) emits conda_bash_env_name, but run.py:189 reads env_dict.get("conda_environment", ""). The bash env path appears to work today only because env=None is always passed (the stub), so the dict is always empty. When the wire-up happens in Step 2 this mismatch will surface immediately — worth resolving then rather than being surprised by it.
