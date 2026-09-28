"""Contract tests: contracts/internal/checksum-reference-lifecycle.md §8, *The
definition write*, and the §6 ``Transformation`` row.

``Transformation._publish_definition`` writes the definition with
``transfer_write()`` and claims it under ``"definition"`` with a non-scratch
claim, **even for a scratch Transformation** (the definition is provenance: a
scratch result can be fingertipped elsewhere only if the definition is readable).
Whether the §1 scratch ruling covers this write is *deferred*: the page records
this as the current behaviour, "neither as a gap nor as settled contract". These
tests pin that recorded behaviour, so that a change to it is noticed; if the
deferred ruling goes the other way, the scratch case must be rewritten.

Contrast (settled, §8 *Buffer persistence*): the scratch Transformation's public
``"result"`` claim is a scratch owner claim and does not publish.
"""

from __future__ import annotations

import uuid

import pytest

from seamless import Buffer, Checksum
from seamless.caching import buffer_writer
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.transformer import delayed


def _tag(token):
    return "defined-" + token


@pytest.fixture
def writes(monkeypatch):
    written: list[Checksum] = []
    monkeypatch.setattr(
        buffer_writer, "register", lambda buf: written.append(buf.get_checksum())
    )
    return written


@pytest.mark.parametrize("scratch", [False, True], ids=["non-scratch", "scratch"])
def test_definition_is_written_and_claimed_non_scratch_even_for_scratch(writes, scratch):
    cache = get_buffer_cache()
    builder = delayed(_tag)
    builder.local = True
    builder.scratch = scratch
    transformation = builder(uuid.uuid4().hex)
    try:
        result = transformation.compute()
        assert result is not None, transformation.exception
        definition = transformation.transformation_checksum
        roles = [role for cs, role in transformation._refheld_checksums() if cs == definition]
        assert roles == ["definition"]
        assert definition in writes, "the definition was not written"
        assert cache.is_scratch_ref(definition) is False
        # The result follows the Transformation's scratch policy (§8).
        assert (result in writes) is (not scratch)
        assert cache.is_scratch_ref(result) is scratch
    finally:
        transformation._release_refholds()
    assert cache.reference_snapshot().get(definition, (0, 0, False))[0] == 0


def test_pretransformation_code_claim_follows_scratch_policy(writes):
    cache = get_buffer_cache()
    builder = delayed(_tag)
    builder.local = True
    builder.scratch = True
    codebuf = builder._snapshot_for_call().codebuf
    if isinstance(codebuf, Buffer):
        code_checksum = codebuf.get_checksum()
    elif isinstance(codebuf, Checksum):
        code_checksum = codebuf
    else:
        code_checksum = Buffer(codebuf, "python").get_checksum()
    cache.mark_scratch(code_checksum)

    transformation = builder(uuid.uuid4().hex)
    try:
        result = transformation.compute()
        assert result is not None, transformation.exception
        assert code_checksum not in writes
        assert cache.is_scratch_ref(code_checksum) is True
    finally:
        transformation._release_refholds()
