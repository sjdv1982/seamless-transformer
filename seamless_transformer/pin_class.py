"""Transformer-owned whole-input handles."""
from seamless import CellBase, Checksum, Expression
from seamless.cell_class import (
    _UNSET, _check_input_ref, _is_input_ref, _typed_input_celltype,
    _serialize_value, _checksum_for_buffer, _available_input_checksum,
)
from .transformation_utils import validate_pin_null


class Pin(CellBase):
    """A fresh handle to a Transformer input; a Pin cannot be a source."""
    __slots__ = ()

    def __init__(self, transformer, name):
        self._workflow_backend = StandalonePinBackend(transformer, name)
        self._refholds_released = True  # The Transformer owns the input claims.

    def build(self, input_ref=_UNSET):
        return self._workflow_backend.build(input_ref)

    def fingertip(self):
        return self._workflow_backend.fingertip()

    @classmethod
    def _from_backend(cls, backend):
        pin = cls.__new__(cls)
        pin._workflow_backend = backend
        pin._refholds_released = True
        return pin


class StandalonePinBackend:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self._memo = owner._pin_memos.setdefault(name, {
            'exception': None, 'checksum': None, 'identity': None,
        })

    _exception = property(lambda self: self._memo['exception'],
                          lambda self, value: self._memo.__setitem__('exception', value))
    _result_identity = property(lambda self: self._memo['identity'],
                               lambda self, value: self._memo.__setitem__('identity', value))

    @property
    def _result_checksum(self):
        return self._memo['checksum']

    @_result_checksum.setter
    def _result_checksum(self, value):
        self.owner._replace_checksum_field(self._memo['checksum'], value)
        self._memo['checksum'] = value

    def _identity(self):
        from seamless import Cell
        from seamless.cell_class import _cell_recipe_key
        ref = self._input_ref
        if isinstance(ref, Cell) and ref._workflow_backend is None:
            source = _cell_recipe_key(ref)
        elif isinstance(ref, Cell):
            checksum = ref.checksum
            source = ('cell', id(ref), checksum.hex() if checksum is not None else None)
        else:
            source = ('checksum', ref.hex()) if isinstance(ref, Checksum) else ('object', id(ref))
        return source, self.input_celltype, self.celltype, self.name in self.owner._optional_pins

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
        self._exception = None
        self._result_checksum = None
        self._result_identity = None

    def build(self, input_ref=_UNSET):
        ref = self._input_ref if input_ref is _UNSET else _check_input_ref(input_ref)
        declared = self.input_celltype if input_ref is _UNSET else _typed_input_celltype(ref) or self.celltype
        return Expression(ref, input_celltype=declared, celltype=self.celltype)

    def _computed(self, checksum, input_ref):
        validate_pin_null(checksum, self.celltype, self.name,
                          optional=self.name in self.owner._optional_pins)
        if input_ref is _UNSET:
            self._result_identity = self._identity()
            self._exception = None
            self._result_checksum = checksum
        return checksum

    def _compute_error(self, exc, input_ref):
        from seamless.error_envelope import execution_error, RunningLoopRefusal
        if isinstance(exc, RunningLoopRefusal):
            return None
        if input_ref is not _UNSET:
            raise exc
        self._result_identity = self._identity()
        self._exception = execution_error(exc)
        self._result_checksum = None
        return None

    def compute(self, input_ref=_UNSET, *, timeout=None):
        try:
            return self._computed(self.build(input_ref).compute(), input_ref)
        except Exception as exc:
            return self._compute_error(exc, input_ref)

    async def compute_async(self, input_ref=_UNSET, *, timeout=None):
        try:
            return self._computed(await self.build(input_ref).compute_async(), input_ref)
        except Exception as exc:
            return self._compute_error(exc, input_ref)

    def run(self, input_ref=_UNSET):
        checksum = self.compute(input_ref)
        if input_ref is _UNSET:
            return self.value
        if checksum is None:
            return None
        value = checksum.resolve(self.celltype)
        return value.content if self.celltype == 'bytes' and hasattr(value, 'content') else value

    @property
    def checksum(self):
        if self._input_ref is None:
            self._exception = None
            self._result_checksum = None
            self._result_identity = None
            return None
        ref = self._input_ref
        identity = self._identity()
        if self._result_identity != identity:
            self._exception = None
            self._result_checksum = None
            self._result_identity = identity
        if self._exception is not None:
            return None
        if self._result_checksum is not None:
            return self._result_checksum
        from seamless.error_envelope import RunningLoopRefusal
        from seamless.checksum.null import canonicalize_checksum, is_null
        try:
            input_checksum = _available_input_checksum(self._input_ref)
            if input_checksum is None:
                return None
            input_checksum = canonicalize_checksum(input_checksum, self.input_celltype)
            validate_pin_null(input_checksum, self.celltype, self.name,
                              optional=self.name in self.owner._optional_pins)
            if self.input_celltype == self.celltype or is_null(input_checksum):
                checksum = input_checksum
            else:
                checksum = Expression(
                    input_checksum,
                    input_celltype=self.input_celltype,
                    celltype=self.celltype,
                ).compute()
        except RunningLoopRefusal:
            return None
        except Exception as exc:
            from seamless.error_envelope import execution_error
            self._exception = execution_error(exc)
            return None
        self._exception = None
        self._result_checksum = checksum
        self._result_identity = identity
        return checksum

    @property
    def state(self):
        if self._input_ref is None:
            return 'unwired'
        checksum = self.checksum
        if checksum is not None:
            return 'complete'
        return 'failed' if self._exception is not None else 'waiting'

    @property
    def exception(self):
        self.checksum
        return str(self._exception) if self._exception is not None else None

    @property
    def buffer(self):
        checksum = self.checksum
        if checksum is None:
            if self._exception is not None:
                raise self._exception
            return None
        try:
            from seamless.checksum.hash_type_validation import validate_deserializable_as
            validate_deserializable_as(checksum, self.celltype)
            buffer = checksum.resolve()
            validate_deserializable_as(checksum, self.celltype, buffer=buffer)
            return buffer
        except Exception as exc:
            return self._materialization_error(exc)

    @property
    def value(self):
        if self._input_ref is None:
            return None
        checksum = self.checksum
        if checksum is None:
            if self._exception is not None:
                raise self._exception
            return None
        try:
            value = checksum.resolve(self.celltype)
        except Exception as exc:
            return self._materialization_error(exc)
        return value.content if self.celltype == 'bytes' and hasattr(value, 'content') else value

    def _materialization_error(self, exc):
        from seamless import CacheMissError
        if isinstance(exc, CacheMissError):
            raise exc
        from seamless.error_envelope import execution_error
        self._exception = execution_error(exc)
        raise self._exception

    def fingertip(self):
        # Recovery must not demand evaluation of the pin's expression.
        ref = self._input_ref
        identity = self._identity()
        checksum = self._result_checksum if self._result_identity == identity else None
        if self._result_identity == identity and self._exception is not None:
            return None
        if checksum is None and isinstance(ref, Checksum) and self.input_celltype == self.celltype:
            from seamless.checksum.null import canonicalize_checksum
            checksum = canonicalize_checksum(ref, self.celltype)
        return None if checksum is None else checksum.fingertip_sync()

    def clear_exception(self):
        self._exception = None

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
        self._result_checksum = None
        self._result_identity = None

    def write_value(self, value, *, detach=False):
        self._check_write_authority(detach)
        # A Checksum is a value exactly when the pin's celltype is checksum.
        checksum_value = isinstance(value, Checksum) and self.celltype == 'checksum'
        if value is not None and not checksum_value and _is_input_ref(value):
            ref = value
            declared = _typed_input_celltype(ref) or self.celltype
            from seamless.cell_class import _check_projected_source
            _check_projected_source(ref, self.celltype)
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
