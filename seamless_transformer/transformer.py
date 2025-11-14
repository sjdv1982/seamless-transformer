"""Wraps a function in a direct transformer
Direct transformers can be called as normal functions, but
the source code of the function and the arguments are converted
into a Seamless transformation."""

from copy import deepcopy
from functools import partial, update_wrapper
import inspect
from seamless import Checksum, Buffer, CacheMissError

"""
from seamless.util.is_forked import is_forked
from seamless.checksum.database_client import database
from seamless.checksum.buffer_remote import has_readwrite_servers
from seamless.checksum.get_buffer import get_buffer
from seamless.checksum.deserialize import deserialize_sync

from .Transformation import Transformation, transformation_from_dict
from ..Environment import Environment
"""


def transformer(
    func=None,
    *,
    scratch=None,
    direct_print=None,
    local=None,
    return_transformation=False,
    in_process=False,
):
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
            in_process=in_process,
        )
    result = Transformer(
        func,
        scratch=scratch,
        direct_print=direct_print,
        local=local,
        return_transformation=return_transformation,
        in_process=in_process,
    )
    update_wrapper(result, func)
    return result


class Transformer:
    """Transformer.
    Transformers can be called as normal functions, but
    the source code of the function and the arguments are converted
    into a Seamless transformation. Doing so imports seamless.workflow."""

    def __init__(
        self, func, *, scratch, direct_print, local, return_transformation, in_process
    ):
        """Transformer.
        Transformers can be called as normal functions, but
        the source code of the function and the arguments are converted
        into a Seamless transformation. Doing so imports seamless.workflow.

        Parameters:

        - local. If True, transformations are executed in the local
                    Seamless instance.
                If False, they are delegated to the assistant, which must exist.
                If None (default), the assistant tried first
                and local execution is a fallback for if there is no assistant.

        - return_transformation.
                If False, calling the function executes it immediately,
                    returning its value.
                If True, it returns a Transformation object.
                Imperative transformations can be queried for their .value
                or .logs. Doing so forces their execution.
                As of Seamless 0.12, forcing one transformation also forces
                    all other transformations.

        - scratch  ...

        - direct_print ...

        - in_process ...

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
        self._in_process = in_process

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
        from .pretransformation import direct_transformer_to_pretransformation
        from .run import run_transformation_dict_in_process
        from .transformation_utils import tf_get_buffer

        """
        from seamless.workflow.core.direct.module import get_module_definition
        """

        """
        STUB
        if is_forked():
            assert not self._in_process
            if not database.active or not has_readwrite_servers():
                raise RuntimeError(
                    # pylint: disable=line-too-long
                    "Running @transformer inside a transformation requires Seamless database and buffer servers"
                )
        /STUB
        """

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

        meta = deepcopy(self._meta)
        result_celltype = self._celltypes["result"]
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
        if self._return_transformation:
            """
            STUB
            tf = transformation_from_dict(
                pre_transformation.pretransformation_dict,
                result_celltype,
                upstream_dependencies=deps,
            )
            tf.scratch = self.scratch
            return tf
            /STUB
            """
            raise NotImplementedError
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
                tf_checksum = tf_buffer.get_checksum()
                ### tf_dunder = extract_dunder(pre_transformation.pretransformation_dict)
                tf_dunder = {}  # TODO

                result_checksum = run_transformation_dict_in_process(
                    pre_transformation.pretransformation_dict,
                    tf_checksum=tf_checksum,
                    tf_dunder=tf_dunder,
                    scratch=self.scratch,
                )
            finally:
                pre_transformation.release()
            result_checksum = Checksum(result_checksum)
            if not result_checksum:
                raise RuntimeError("Result is empty")
            buf = result_checksum.resolve()
            if result_celltype in ("deepcell", "folder"):
                result_celltype = "plain"
            return buf.get_value(result_celltype)

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
