"""High-level structure for preparing transformer dicts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from seamless import Buffer, Checksum

from .code_manager import CodeManager, get_code_manager


NON_CHECKSUM_ITEMS = (
    "__output__",
    "__language__",
    "__meta__",
    "__env__",
    "__format__",
)


class PreTransformation:
    """Wrapper around a pre-transformation dict that can be prepared incrementally."""

    def __init__(
        self,
        pretransformation_dict: Dict[str, Any],
        *,
        code_manager: Optional[CodeManager] = None,
    ):
        self._pretransformation_dict = pretransformation_dict
        self._code_manager = code_manager or get_code_manager()
        self._prepared = False
        self._code_refs: list[tuple[Checksum, Checksum]] = []

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

    def release(self) -> None:
        """Release code-manager references."""
        if not self._code_refs:
            return
        for semantic_checksum, syntactic_checksum in self._code_refs:
            self._code_manager.decref_syntactic(syntactic_checksum)
            self._code_manager.decref_semantic(semantic_checksum)
        self._code_refs.clear()

    def __del__(self):
        try:
            self.release()
        except Exception:
            # Suppress destructor errors
            pass

    # --- helpers --------------------------------------------------------------
    def _prepare_pin_value(self, argname: str, value, celltype: str):
        if argname == "code":
            return self._prepare_code(value)
        checksum = self._to_checksum(value, celltype)
        return checksum

    def _prepare_code(self, value) -> Checksum:
        code_buffer = value if isinstance(value, Buffer) else Buffer(value, "python")
        semantic_checksum, syntactic_checksum = self._code_manager.track_code_buffer(
            code_buffer
        )
        self._code_manager.incref_syntactic(syntactic_checksum)
        self._code_manager.incref_semantic(semantic_checksum)
        self._code_refs.append((semantic_checksum, syntactic_checksum))
        self._pretransformation_dict["__code_checksum__"] = syntactic_checksum.hex()
        return semantic_checksum

    @staticmethod
    def _to_checksum(value, celltype: str) -> Checksum | None:
        if value is None:
            return None
        if isinstance(value, Checksum):
            return value
        if isinstance(value, str) and len(value) == 64:
            return Checksum(value)
        if isinstance(value, Buffer):
            buffer = value
        else:
            buffer = Buffer(value, celltype or "mixed")
        return buffer.get_checksum()


def direct_transformer_to_pretransformation(
    codebuf,
    meta,
    celltypes,
    modules,
    arguments,
    env,
    *,
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
        "__language__": "python",
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

    tf_pins["code"] = {"celltype": "python", "subcelltype": "transformer"}
    arguments2["code"] = codebuf

    format_section: Dict[str, Dict[str, Any]] = {}
    for pinname, pin in tf_pins.items():
        if pinname == "result":
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
