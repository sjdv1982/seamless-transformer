"""Minimal injector implementation ported for seamless_transformer."""

import sys
from types import ModuleType
from contextlib import contextmanager


class Package:
    """Placeholder Package type used by Injector.

    The full implementation is part of the legacy Seamless codebase and will be
    ported later. For now, we only need the type check.
    """

    def __init__(self, mapping=None):
        self.mapping = mapping or {}


class Injector:
    """Injects dynamically built modules into sys.modules."""

    def __init__(self, topmodule_name):
        self.topmodule_name = topmodule_name
        self.topmodule = ModuleType(topmodule_name)
        self.topmodule.__package__ = topmodule_name
        self.topmodule.__path__ = []

    @contextmanager
    def active_workspace(self, workspace, namespace):
        sys_modules = sys.modules
        old_modules = {}
        old_packages = {}
        old_names = {}
        new_modules = {}
        if self.topmodule_name in sys_modules:
            old_modules[self.topmodule_name] = sys_modules[self.topmodule_name]
        try:
            sys_modules[self.topmodule_name] = self.topmodule
            package_mapping = {}
            for modname, mod in workspace.items():
                if isinstance(mod, Package):
                    for k, v in mod.mapping.items():
                        if v == "__init__":
                            vv = modname
                        else:
                            vv = modname + "." + v
                        package_mapping[k] = vv
                        assert k in workspace, k
            for modname, mod in workspace.items():
                if isinstance(mod, Package):
                    continue
                modname2 = modname
                is_abs = False
                if modname in package_mapping:
                    modname2 = package_mapping[modname]
                elif modname.find(".") > -1:
                    # absolute module name injection
                    is_abs = True
                if is_abs:
                    continue
                if self.topmodule_name:
                    mname = self.topmodule_name + "." + modname2
                else:
                    mname = modname2
                if mname.endswith(".__init__"):
                    mname = mname[: -len(".__init__")]
                if mname in sys_modules:
                    old_modules[mname] = sys_modules[mname]
                new_modules[modname] = mname
                sys_modules[mname] = mod
                if mod.__name__ in sys_modules:
                    old_modules[mod.__name__] = sys_modules[mod.__name__]
                sys_modules[mod.__name__] = mod
                namespace[modname2] = mod
                old_packages[mname] = getattr(mod, "__package__", None)
                old_names[mname] = getattr(mod, "__name__", None)
                package_name = mname
                if package_name.endswith(".__init__"):
                    package_name = package_name[: -len(".__init__")]
                else:
                    pos = package_name.rfind(".")
                    if pos > -1:
                        package_name = package_name[:pos]
                mod.__package__ = package_name
                mod.__name__ = mname
                mod.__path__ = []
            yield
        finally:
            if self.topmodule_name in old_modules:
                sys_modules[self.topmodule_name] = old_modules[self.topmodule_name]
            else:
                sys_modules.pop(self.topmodule_name, None)
            for modname, mod in workspace.items():
                if modname not in new_modules:
                    continue
                mname = new_modules[modname]
                if mname in old_packages:
                    assert mname in sys_modules, mname
                    mod = sys_modules[mname]
                    mod.__package__ = old_packages[mname]
                if mname in old_names:
                    assert mname in sys_modules, mname
                    mod = sys_modules[mname]
                    mod.__name__ = old_names[mname]

            for modname, mod in workspace.items():
                if modname not in new_modules:
                    continue
                mname = new_modules[modname]
                if mname in old_modules:
                    sys_modules[mname] = old_modules[mname]
                else:
                    sys_modules.pop(mname, None)
                if mod.__name__ in old_modules:
                    sys_modules[mod.__name__] = old_modules[mod.__name__]
                else:
                    sys_modules.pop(mod.__name__, None)


transformer_injector = Injector("transformer")

__all__ = ["Injector", "transformer_injector", "Package"]
