"""Live, read-only optional-pin views derived from Python signatures."""
import inspect
from collections.abc import Set


def pin_signature(signature):
    """Variadic arguments do not declare pins."""
    if signature is None:
        return None
    return signature.replace(parameters=[
        p for p in signature.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ])


def optional_names(signature):
    signature = pin_signature(signature)
    return set() if signature is None else {
        name for name, p in signature.parameters.items()
        if p.default is not inspect.Parameter.empty
    }


class OptionalPin:
    __slots__ = ('_pins', '_name')

    def __init__(self, pins, name):
        self._pins, self._name = pins, name

    def disable(self):
        self._pins._toggle(self._name, False)

    def enable(self):
        self._pins._toggle(self._name, True)


class OptionalPins(Set):
    """Enabled names iterate as a set; disabled names remain discoverable."""
    __slots__ = ('_owner',)

    def __init__(self, owner):
        self._owner = owner

    def _names(self):
        if self._owner.language != 'python':
            return set()
        return optional_names(self._owner._get_signature())

    def _enabled(self):
        backend = self._owner._workflow_backend
        enabled = self._owner._optional_pins if backend is None else backend.cfg.optional_pins
        return self._names() & enabled

    def __contains__(self, name):
        return name in self._enabled()

    def __iter__(self):
        return iter(self._enabled())

    def __len__(self):
        return len(self._enabled())

    def __dir__(self):
        return sorted(set(super().__dir__()) | self._names())

    def __getattr__(self, name):
        return self[name]

    def __getitem__(self, name):
        if name not in self._names():
            raise AttributeError(name)
        return OptionalPin(self, name)

    def __repr__(self):
        return repr(self._enabled())

    @classmethod
    def _from_iterable(cls, values):
        return frozenset(values)

    def _toggle(self, name, enabled):
        if name not in self._names():
            raise AttributeError(name)
        backend = self._owner._workflow_backend
        if backend is not None:
            backend.context._set_node_config(backend.node_path, 'optional_pin', enabled, key=name)
        elif enabled:
            self._owner._optional_pins.add(name)
        else:
            self._owner._optional_pins.discard(name)
