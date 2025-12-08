"""Wraps a function in a direct transformer
Direct transformers can be called as normal functions, but
the source code of the function and the arguments are converted
into a Seamless transformation."""

from copy import deepcopy
from functools import partial, update_wrapper
import inspect
import pickle
from seamless import Checksum, Buffer, CacheMissError, ensure_open
from .pretransformation import direct_transformer_to_pretransformation
from .transformation_class import transformation_from_pretransformation
from .transformation_cache import run_sync
from .transformation_utils import unpack_deep_structure, is_deep_celltype, tf_get_buffer
from . import worker
import multiprocessing as mp
import sys


def transformer(
    func=None,
    *,
    scratch=False,
    direct_print=False,
    local=False,
    return_transformation=False,
) -> "Transformer":
    """Wraps a function in a direct transformer
    Direct transformers can be called as normal functions, but
    the source code of the function and the arguments are converted
    into a Seamless transformation."""
    if func is None:
        return partial(
            transformer,
            scratch=scratch,
            direct_print=direct_print,
            local=local,
            return_transformation=return_transformation,
        )
    result = Transformer(
        func,
        scratch=scratch,
        direct_print=direct_print,
        local=local,
        return_transformation=return_transformation,
    )
    update_wrapper(result, func)
    return result


class Transformer:
    """Transformer.
    Transformers can be called as normal functions, but
    the source code of the function and the arguments are converted
    into a Seamless transformation. Doing so imports seamless.workflow."""

    def __init__(self, func, *, scratch, direct_print, local, return_transformation):
        """Transformer.
        Transformers can be called as normal functions, but
        the source code of the function and the arguments are converted
        into a Seamless transformation. Doing so imports seamless.workflow.

        Parameters:

        - local. If True, transformations are executed in the local
                    Seamless instance.
                If False (default), they are executed remotely if possible.

        - return_transformation.
                If False (default), calling the function executes it immediately,
                    returning its value.
                If True, it returns a Transformation object.

        - scratch.
                If True, only the checksum is preserved. This is for cases where
                  the result is bulky, but can be recomputed easily
                If False (default), the buffers are preserved, so that the value
                  can be accessed if needed.


        - direct_print: If True, it is attempted to print stdout and stderr
                while the transformation runs.

        Attributes:

        - meta. Accesses all meta-information (including local)

        - celltypes. Returns a wrapper where you can set the celltypes
                of the individual transformer pins.
            The syntax is: Transformer.celltypes.a = "text"
            (or Transformer.celltypes["a"] = "text")
            for pin "a".

        - modules: Returns a wrapper where you can define Python modules
            to be imported into the transformation

        - environment  ...

        """
        from .getsource import getsource

        code = getsource(func)
        codebuf = Buffer(code, "python")

        signature = inspect.signature(func)
        self._return_transformation = return_transformation
        self._signature = signature
        self._codebuf = codebuf
        self._celltypes = {k: "mixed" for k in signature.parameters}
        self._celltypes["result"] = "mixed"
        self._modules = {}
        """
        STUB
        self._environment = Environment(self)
        self._environment_state = None
        """
        self._meta = {"transformer_path": ["tf", "tf"], "local": local}
        self.scratch = scratch
        self.direct_print = direct_print
        update_wrapper(self, func)

    @property
    def celltypes(self):
        """The celltypes"""
        return CelltypesWrapper(self._celltypes)

    @property
    def modules(self):
        """The imported modules"""
        return ModulesWrapper(self._modules)

    '''
    STUB
    @property
    def environment(self) -> "Environment":
        """Computing environment to execute transformations in"""
        return self._environment
    /STUB
    '''

    def __call__(self, *args, **kwargs):
        """
        from seamless.workflow.core.direct.module import get_module_definition
        """

        ensure_open("transformer call")
        arguments = self._signature.bind(*args, **kwargs).arguments
        deps = {}
        """
        STUB
        for argname, arg in arguments.items():
            if isinstance(arg, Transformation):
                deps[argname] = arg

        env = self._environment._to_lowlevel()
        /STUB
        """
        env = None

        result_celltype = self.celltypes["result"]
        meta = deepcopy(self._meta)
        modules = {}
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

        pre_transformation = direct_transformer_to_pretransformation(
            self._codebuf, meta, self._celltypes, modules, arguments, env
        )
        tf_dunder = {"globals": self._collect_execution_globals()}
        if self._return_transformation:
            return transformation_from_pretransformation(
                pre_transformation,
                upstream_dependencies=deps,
                meta=meta,
                scratch=self.scratch,
            )
        else:
            for depname, dep in deps.items():
                dep.compute()
                if dep.exception is not None:
                    msg = "Dependency '{}' has an exception: {}"
                    raise RuntimeError(msg.format(depname, dep.exception))

            try:
                pre_transformation.prepare_transformation()

                ### increfed, tf_checksum = register_transformation_dict(pre_transformation.pretransformation_dict,)
                tf_buffer = tf_get_buffer(pre_transformation.pretransformation_dict)
                tf_buffer.tempref()
                tf_checksum = tf_buffer.get_checksum()
                result_checksum = run_sync(
                    pre_transformation.pretransformation_dict,
                    tf_checksum=tf_checksum,
                    tf_dunder=tf_dunder,
                    scratch=self.scratch,
                    require_fingertip=True,
                )
            finally:
                pre_transformation.release()
            result_checksum = Checksum(result_checksum)
            if not result_checksum:
                raise RuntimeError("Result is empty")
            buf = result_checksum.resolve()
            assert isinstance(buf, Buffer)
            target_celltype = result_celltype
            if result_celltype == "folder":
                target_celltype = "plain"
            value = buf.get_value(target_celltype)
            if is_deep_celltype(result_celltype):
                value = unpack_deep_structure(value, result_celltype)
            return value

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

    @property
    def return_transformation(self) -> bool:
        """If True, calling this Transformer returns a Transformation object.
        Otherwise, calling causes a direct transformation execution, returning its result.
        """
        return self._return_transformation

    @return_transformation.setter
    def return_transformation(self, value: bool):
        self._return_transformation = value

    def copy(
        self,
        scratch=None,
        direct_print=None,
        local=None,
        return_transformation=False,
    ):
        """Make a copy of the Transformer.

        If other parameters are provided, their attributes on the copy
        are set to the provided value.
        """
        transformer = deepcopy(self)
        if scratch is not None:
            transformer.scratch = scratch
        if direct_print is not None:
            transformer.direct_print = direct_print
        if local is not None:
            transformer.local = local
        if return_transformation is not None:
            transformer.return_transformation = return_transformation
        return transformer

    def _collect_execution_globals(self) -> dict[str, object]:
        func = getattr(self, "__wrapped__", None)
        if func is None or not hasattr(func, "__code__"):
            return {}
        names = set(func.__code__.co_freevars) | set(func.__code__.co_names)
        closure_cells = dict(zip(func.__code__.co_freevars, func.__closure__ or []))
        globals_map: dict[str, object] = {}
        for name in names:
            value = None
            if name in closure_cells:
                try:
                    value = closure_cells[name].cell_contents
                except ValueError:
                    continue
            elif name in func.__globals__:
                value = func.__globals__[name]
            if value is None:
                continue
            try:
                pickle.dumps(value)
                globals_map[name] = value
                continue
            except Exception:
                pass
            if isinstance(value, Transformer):
                if worker.has_spawned():
                    try:
                        globals_map[name] = worker.register_transformer_proxy(value)
                        continue
                    except Exception:
                        pass
                globals_map[name] = value
        return globals_map


class CelltypesWrapper:
    """Wrapper around an imperative transformer's celltypes."""

    def __init__(self, celltypes):
        self._celltypes = celltypes

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
            raise AttributeError(key)
        if key == "result":
            if value in ("deepfolder", "module"):
                raise TypeError(f"result pin celltype cannot be '{value}'")
            pin_celltypes = celltypes + ["deepcell", "folder"]
        else:
            pin_celltypes = celltypes + ["deepcell", "deepfolder", "folder", "module"]
        if value not in pin_celltypes:
            raise TypeError(value, pin_celltypes)
        self._celltypes[key] = value

    def __dir__(self):
        return sorted(self._celltypes.keys())

    def __str__(self):
        return str(self._celltypes)

    def __repr__(self):
        return str(self)


class ModulesWrapper:
    """Wrapper around an imperative transformer's imported modules.

    Modifying this wrapper imports seamless.workflow"""

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
            if not isinstance(value, ModuleType):
                raise TypeError(type(value))
            self._modules[attr] = value

    def __dir__(self):
        return sorted(self._modules.keys())

    def __str__(self):
        return str(self._modules)

    def __repr__(self):
        return str(self)
