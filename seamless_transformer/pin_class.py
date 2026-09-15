"""Transformer-owned whole-input handles."""
from seamless import CellBase, Checksum, Expression
from seamless.cell_class import (
    _UNSET, _check_input_ref, _is_input_ref, _typed_input_celltype,
    _serialize_value, _checksum_for_buffer,
)
from .transformation_utils import validate_pin_null


class Pin(CellBase):
    """A fresh handle to a Transformer input; a Pin cannot be a source."""
    __slots__ = ()

    def __init__(self, transformer, name):
        self._workflow_backend = StandalonePinBackend(transformer, name)
        self._refholds_released = True  # The Transformer owns the input claims.

    @classmethod
    def _from_backend(cls, backend):
        pin = cls.__new__(cls)
        pin._workflow_backend = backend
        pin._refholds_released = True
        return pin


class StandalonePinBackend:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self._exception = None

    def _entry(self):
        if self.name == 'result' or self.name not in self.owner._celltypes:
            raise AttributeError(self.name)
        return self.owner._args.get(self.name, (None, None))

    @property
    def _input_ref(self):
        return self._entry()[0]

    @property
    def source(self):
        ref = self._input_ref
        return None if isinstance(ref, Checksum) else ref

    @property
    def input_celltype(self):
        ref, declared = self._entry()
        return _typed_input_celltype(ref) or declared

    @property
    def celltype(self):
        self._entry()
        return self.owner.celltypes[self.name]

    @celltype.setter
    def celltype(self, value):
        self._entry()
        self.owner.celltypes[self.name] = value

    def build(self, input_ref=_UNSET):
        ref = self._input_ref if input_ref is _UNSET else _check_input_ref(input_ref)
        declared = self.input_celltype if input_ref is _UNSET else _typed_input_celltype(ref) or self.celltype
        return Expression(ref, input_celltype=declared, celltype=self.celltype)

    def compute(self, input_ref=_UNSET, *, timeout=None):
        checksum = self.build(input_ref).compute()
        validate_pin_null(checksum, self.celltype, self.name,
                          optional=self.name in self.owner._optional_pins)
        return checksum

    async def compute_async(self, input_ref=_UNSET, *, timeout=None):
        checksum = await self.build(input_ref).compute_async()
        validate_pin_null(checksum, self.celltype, self.name,
                          optional=self.name in self.owner._optional_pins)
        return checksum

    def run(self, input_ref=_UNSET):
        checksum = self.compute(input_ref)
        value = checksum.resolve(self.celltype)
        return value.content if self.celltype == 'bytes' and hasattr(value, 'content') else value

    @property
    def checksum(self):
        if self._input_ref is None:
            self._exception = None
            return None
        try:
            checksum = self.compute()
        except Exception as exc:
            self._exception = exc
            return None
        self._exception = None
        return checksum

    @property
    def state(self):
        if self._input_ref is None:
            return 'unwired'
        return 'complete' if self.checksum is not None else 'failed'

    @property
    def exception(self):
        self.checksum
        return self._exception

    @property
    def buffer(self):
        checksum = self.checksum
        return None if checksum is None else checksum.resolve()

    @property
    def value(self):
        if self._input_ref is None:
            return None
        return self.run()

    def _check_write_authority(self, detach):
        self._entry()
        if not detach and self.source is not None:
            from seamless import AuthorityError
            raise AuthorityError('The pin is controlled by a source; assign .value, .buffer or .checksum to replace it')
    def _replace(self, ref, declared, detach):
        self._check_write_authority(detach)
        old, _ = self._entry()
        self.owner._replace_checksum_field(old, ref)
        if ref is None:
            self.owner._args.pop(self.name, None)
        else:
            self.owner._args[self.name] = (ref, declared)
        self._exception = None

    def write_value(self, value, *, detach=False):
        self._check_write_authority(detach)
        # A Checksum is a value exactly when the pin's celltype is checksum.
        checksum_value = isinstance(value, Checksum) and self.celltype == 'checksum'
        if value is not None and not checksum_value and _is_input_ref(value):
            ref = value
            declared = _typed_input_celltype(ref) or self.celltype
        else:
            ref = _serialize_value(value, self.celltype)
            declared = self.celltype
        if isinstance(ref, Checksum):
            validate_pin_null(ref, self.celltype, self.name,
                              optional=self.name in self.owner._optional_pins)
        self._replace(ref, declared, detach)

    def write_checksum(self, checksum, *, input_celltype=None, detach=False):
        self._check_write_authority(detach)
        ref = None if checksum is None else Checksum(checksum)
        declared = input_celltype or self.celltype if ref is not None else None
        validate_pin_null(ref, self.celltype, self.name,
                          optional=self.name in self.owner._optional_pins)
        self._replace(ref, declared, detach)

    def write_buffer(self, buffer, *, detach=False):
        self._check_write_authority(detach)
        self.write_checksum(_checksum_for_buffer(buffer, self.celltype), detach=detach)
