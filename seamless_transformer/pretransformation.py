"""High-level structure for preparing transformer dicts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from seamless import Buffer, Checksum, is_worker

from .code_manager import CodeManager, get_code_manager


NON_CHECKSUM_ITEMS = (
    "__output__",
    "__language__",
    "__meta__",
    "__env__",
    "__format__",
    "__code_text__",
    "__code_checksum__",
)


class PreTransformation:
    """Wrapper around a pre-transformation dict that can be prepared incrementally."""

    def __init__(
        self,
        pretransformation_dict: Dict[str, Any],
        *,
        code_manager: Optional[CodeManager] = None,
    ):
        if "__language__" not in pretransformation_dict:
            raise ValueError("pretransformation dict must include __language__")
        self._pretransformation_dict = pretransformation_dict
        self._code_manager = code_manager or get_code_manager()
        self._prepared = False
        self._code_refs: list[tuple[Checksum, Checksum]] = []
        self._value_refs: list[Checksum] = []

    @property
    def pretransformation_dict(self) -> Dict[str, Any]:
        """Return the underlying pre-transformation dict."""
        return self._pretransformation_dict

    @property
    def prepared(self) -> bool:
        """True when `prepare_transformation` has run."""
        return self._prepared

    def prepare_transformation(self) -> Dict[str, Any]:
        """Prepare all pins by ensuring they reference checksums."""
        if self._prepared:
            return self._pretransformation_dict

        for argname in list(self._pretransformation_dict.keys()):
            if argname in NON_CHECKSUM_ITEMS:
                continue
            celltype, subcelltype, value = self._pretransformation_dict[argname]
            prepared_value = self._prepare_pin_value(argname, value, celltype)
            if isinstance(prepared_value, Checksum):
                prepared_value = prepared_value.hex()
            self._pretransformation_dict[argname] = (
                celltype,
                subcelltype,
                prepared_value,
            )

        self._prepared = True
        return self._pretransformation_dict

    @property
    def result_celltype(self) -> str:
        return self._pretransformation_dict["__output__"][1]

    def release(self) -> None:
        """Release code-manager references."""
        for semantic_checksum, syntactic_checksum in self._code_refs:
            self._code_manager.decref_syntactic(syntactic_checksum)
            self._code_manager.decref_semantic(semantic_checksum)
        self._code_refs.clear()

        for checksum in self._value_refs:
            checksum.decref()
        self._value_refs.clear()

    def __del__(self):
        try:
            self.release()
        except Exception:
            # Suppress destructor errors
            pass

    def build_partial_transformation(
        self, upstream_dependencies: Optional[Dict[str, "Transformation"]] = None
    ) -> tuple[Dict[str, Any], Dict[str, "Transformation"]]:
        """Prepare a transformation dict, keeping upstream transformations unresolved.

        Non-transformation inputs are converted to checksums, while dependencies are
        left as placeholders so they can be resolved lazily by a Dask client.
        """
        from .transformation_class import Transformation

        tf_dict: Dict[str, Any] = {}
        dependencies: Dict[str, Transformation] = {}
        upstream_dependencies = upstream_dependencies or {}
        for argname in list(self._pretransformation_dict.keys()):
            raw_value = self._pretransformation_dict[argname]
            if argname in NON_CHECKSUM_ITEMS:
                tf_dict[argname] = raw_value
                continue
            celltype, subcelltype, value = raw_value
            prepared_value = self._prepare_pin_value_for_dask(argname, value, celltype)
            if isinstance(prepared_value, Transformation):
                dependency = upstream_dependencies.get(argname, prepared_value)
                dependencies[argname] = dependency
                tf_dict[argname] = (celltype, subcelltype, None)
                continue
            if isinstance(prepared_value, Checksum):
                prepared_value = prepared_value.hex()
            tf_dict[argname] = (celltype, subcelltype, prepared_value)
        if "__code_checksum__" in self._pretransformation_dict:
            tf_dict["__code_checksum__"] = self._pretransformation_dict[
                "__code_checksum__"
            ]
        return tf_dict, dependencies

    # --- helpers --------------------------------------------------------------
    def _prepare_pin_value(self, argname: str, value, celltype: str):
        # Convert upstream Transformation dependencies into their result checksum.
        # Dependencies are ensured to have been computed earlier.
        from .transformation_class import Transformation

        if isinstance(value, Transformation):
            if value.exception is not None:
                msg = f"Dependency '{argname}' has an exception:\n{value.exception}"
                raise RuntimeError(msg)
            return value.result_checksum
        if argname == "code":
            if self._pretransformation_dict.get("__language__") == "python":
                return self._prepare_code(value)
            return self._to_checksum(value, celltype)
        checksum = self._to_checksum(value, celltype)
        return checksum

    def _prepare_pin_value_for_dask(self, argname: str, value, celltype: str):
        """Like `_prepare_pin_value` but leaves dependencies unresolved."""
        from .transformation_class import Transformation

        if isinstance(value, Transformation):
            return value
        return self._prepare_pin_value(argname, value, celltype)

    def _prepare_code(self, value) -> Checksum:
        # Allow passing a checksum hex / Checksum pointing to a code buffer.
        if isinstance(value, (Checksum, str, bytes)) and not isinstance(value, Buffer):
            try:
                if isinstance(value, str) and len(value) != 64:
                    raise ValueError
                cs = Checksum(value)
                buf = cs.resolve()
                if isinstance(buf, Buffer):
                    value = buf
            except Exception:
                pass
        code_buffer = value if isinstance(value, Buffer) else Buffer(value, "python")
        if is_worker():
            try:
                code_buffer.tempref()  # upload to parent so nested workers can resolve
            except Exception:
                pass
        try:
            # Keep a plain-text fallback in case the code buffer cannot be resolved later.
            self._pretransformation_dict.setdefault(
                "__code_text__", code_buffer.decode()
            )
        except Exception:
            pass
        semantic_checksum, syntactic_checksum = self._code_manager.track_code_buffer(
            code_buffer
        )
        self._code_manager.incref_syntactic(syntactic_checksum)
        self._code_manager.incref_semantic(semantic_checksum)
        self._code_refs.append((semantic_checksum, syntactic_checksum))
        self._pretransformation_dict["__code_checksum__"] = syntactic_checksum.hex()
        # Prefer syntactic checksum for execution; semantic guard remains tracked.
        return syntactic_checksum

    def _to_checksum(self, value, celltype: str) -> Checksum | None:
        if value is None:
            return None
        if isinstance(value, Checksum):
            checksum = value
        elif isinstance(value, str) and len(value) == 64:
            checksum = Checksum(value)
        else:
            buffer = (
                value
                if isinstance(value, Buffer)
                else Buffer(value, celltype or "mixed")
            )
            checksum = buffer.get_checksum()
            if is_worker():
                try:
                    buffer.tempref()  # ensure parent sees worker-created buffers
                except Exception:
                    pass
        if not is_worker():
            checksum.incref()
        self._value_refs.append(checksum)
        return checksum


class PreparedPreTransformation(PreTransformation):
    """PreTransformation that treats pin values as checksums.

    This avoids re-processing code pins for already-prepared transformation dicts.
    """

    def _prepare_pin_value(self, argname: str, value, celltype: str):
        from .transformation_class import Transformation

        if isinstance(value, Transformation):
            if value.exception is not None:
                msg = f"Dependency '{argname}' has an exception:\n{value.exception}"
                raise RuntimeError(msg)
            return value.result_checksum
        return self._to_checksum(value, celltype)


def direct_transformer_to_pretransformation(
    codebuf,
    meta,
    celltypes,
    modules,
    arguments,
    env,
    *,
    language,
    code_manager: Optional[CodeManager] = None,
) -> PreTransformation:
    """Create a PreTransformation instance for a direct transformer call."""
    result_celltype = celltypes["result"]
    if result_celltype == "folder":
        result_celltype = "deepfolder"
    if result_celltype == "structured":
        result_celltype = "mixed"

    outputpin = ("result", result_celltype, None)

    pretransformation_dict: Dict[str, Any] = {
        "__output__": outputpin,
        "__language__": language,
    }

    if env is not None:
        envbuf = Buffer(env, "plain")
        checksum = envbuf.get_checksum()
        pretransformation_dict["__env__"] = checksum.hex()

    if meta:
        pretransformation_dict["__meta__"] = deepcopy(meta)

    tf_pins: Dict[str, Dict[str, Any]] = {}
    arguments2 = arguments.copy()
    for pinname, module_code in modules.items():
        arguments2[pinname] = module_code
        pin = {"celltype": "plain", "subcelltype": "module"}
        tf_pins[pinname] = pin

    for pinname, celltype in celltypes.items():
        original_celltype = celltype
        if celltype is None or celltype == "default":
            if pinname.endswith("_SCHEMA"):
                celltype = "plain"
            else:
                celltype = "mixed"
        if celltype == "folder":
            celltype = "deepfolder"

        if celltype in ("cson", "yaml", "python"):
            raise NotImplementedError(pinname)

        if celltype == "structured":
            celltype = "mixed"
        if celltype == "module":
            pin = {"celltype": "plain", "subcelltype": "module"}
        elif celltype == "deepfolder":
            filesystem = {
                "mode": "directory",
                "optional": original_celltype == "folder",
            }
            pin = {"celltype": "deepfolder", "filesystem": filesystem}
        elif celltype == "checksum":
            pin = {"celltype": "plain", "subcelltype": "checksum"}
        else:
            pin = {"celltype": celltype}
        tf_pins[pinname] = pin

    code_celltype = "python" if language == "python" else "text"
    tf_pins["code"] = {"celltype": code_celltype, "subcelltype": "transformer"}
    arguments2["code"] = codebuf

    format_section: Dict[str, Dict[str, Any]] = {}
    for pinname, pin in tf_pins.items():
        if pinname == "result":
            continue
        if pinname not in arguments2:
            # optional arg
            continue
        value = arguments2[pinname]
        pretransformation_dict[pinname] = (
            pin["celltype"],
            pin.get("subcelltype"),
            value,
        )
        filesystem = pin.get("filesystem")
        if filesystem is not None:
            format_section.setdefault(pinname, {})
            format_section[pinname]["filesystem"] = deepcopy(filesystem)

    if format_section:
        pretransformation_dict["__format__"] = format_section

    return PreTransformation(pretransformation_dict, code_manager=code_manager)


__all__ = [
    "PreTransformation",
    "direct_transformer_to_pretransformation",
]
