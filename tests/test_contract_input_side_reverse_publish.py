"""Contract tests: a transformer's inputs reverse-publish
(contracts/internal/checksum-reference-lifecycle.md, §1, ruling 2026-09-30).

Non-scratch pins and their Expression inputs are input-side owners. When resolving one
dispatches an Expression (its input is on the hashserver only), the request
carries scratch=False, so the executing side writes the result to the
hashserver, where a transformation running anywhere can find it. A recorded
checksum without reachable bytes does not answer the value request.
"""
import os
import subprocess
import sys
import textwrap
import uuid

import pytest

from test_run_transformation_cli import _write_remote_config

_PRELUDE = '''
import asyncio
import uuid
import seamless
from seamless import Buffer, Cell, Checksum, Expression
import seamless_config
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.checksum import expression as expression_mod
from seamless.checksum.cached_calculate_checksum import checksum_cache
from seamless.transformer import delayed
from seamless_remote import buffer_remote

def drop_buffer(checksum):
    checksum = Checksum(checksum)
    cache = get_buffer_cache()
    with cache.lock:
        cache.weak_cache.pop(checksum, None)
        cache.strong_cache.pop(checksum, None)
    checksum_cache.pop(checksum, None)
    expression_mod._expression_result_buffers.pop(checksum, None)

def hashserver_only(value, celltype):
    buffer = Buffer(value, celltype)
    checksum = buffer.get_checksum()
    assert asyncio.run(buffer_remote.write_buffer(checksum, buffer))
    drop_buffer(checksum)
    return checksum

def on_hashserver(checksum):
    length = asyncio.run(buffer_remote.get_buffer_lengths([checksum]))[0]
    # A read server answers /has with a boolean; a read folder with a length.
    if isinstance(length, bool):
        return length
    return isinstance(length, int) and length >= 0

def echo(value):
    return value + "!"
'''


def _run(tmp_path, body):
    project = 'input-reverse-publish-' + uuid.uuid4().hex
    _write_remote_config(tmp_path, backend='jobserver', project=project)
    script = _PRELUDE + textwrap.dedent(body)
    proc = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                          capture_output=True, text=True, timeout=180,
                          env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'CONTRACT_OK' in proc.stdout


def test_dispatched_expression_input_is_written_to_the_hashserver(tmp_path):
    _run(tmp_path, '''
        seamless_config.init()
        buffer_remote._read_folders_clients.clear()
        try:
            word = "input-" + uuid.uuid4().hex
            source = hashserver_only({"a": word}, "plain")
            expr = Expression(source, "a", input_celltype="plain", celltype="str")
            expected = Buffer(word, "str").get_checksum()
            drop_buffer(expected)
            builder = delayed(echo)
            builder.celltypes.value = "str"
            builder.allow_input_fingertip = False
            transformation = builder(value=expr)
            assert transformation.run() == word + "!"
            assert on_hashserver(expected), "the dispatched input was not written"
        finally:
            seamless.close()
        print("CONTRACT_OK")
    ''')


def test_dispatched_pin_conversion_is_written_to_the_hashserver(tmp_path):
    _run(tmp_path, '''
        seamless_config.init()
        buffer_remote._read_folders_clients.clear()
        try:
            word = "pin-" + uuid.uuid4().hex
            cell = Cell("str", checksum=hashserver_only(word, "str"))
            expected = Buffer(word, "text").get_checksum()
            drop_buffer(expected)
            tf = delayed(echo)
            tf.celltypes.value = "text"
            tf.pins.value = cell  # str -> text needs the buffer: dispatched
            assert tf.pins.value.checksum == expected
            assert on_hashserver(expected), "the dispatched pin conversion was not written"
            assert tf().run() == word + "!"
        finally:
            seamless.close()
        print("CONTRACT_OK")
    ''')
