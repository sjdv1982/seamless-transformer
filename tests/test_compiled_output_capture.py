"""Phase G: native output belongs to the compiled request in worker children."""
import ctypes
import io
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from seamless_transformer import run
from seamless_transformer.output_routing import install_output_routing, output_route

pytestmark = pytest.mark.skipif(not shutil.which("gcc"), reason="gcc required")
SCHEMA = "inputs: []\noutputs:\n  - {name: result, dtype: int32}\n"


@pytest.fixture
def compiled_call(tmp_path, monkeypatch):
    from cffi import FFI
    from seamless_signature import Signature, generate_header
    import yaml
    import seamless_transformer.compiler as compiler

    source = tmp_path / "output.c"
    library = tmp_path / "output.so"
    source.write_text('''#include <stdio.h>
#include <stdint.h>
int emit(int tag, int32_t *result) {
    printf("native-out-%d", tag);
    fprintf(stderr, "native-err-%d", tag);
    *result = tag;
    return 0;
}
int emit_utf8(int32_t *result) {
    const unsigned char euro[] = {0xe2, 0x82, 0xac};
    for (int i = 0; i < 1000; i++) {
        fwrite(euro, 1, sizeof(euro), stdout);
        fwrite(euro, 1, sizeof(euro), stderr);
    }
    *result = 5;
    return 0;
}
int emit_invalid(int32_t *result) {
    const unsigned char text[] = {'a', 0xff, 'b'};
    fwrite(text, 1, sizeof(text), stdout);
    fwrite(text, 1, sizeof(text), stderr);
    *result = 6;
    return 0;
}
int transform(int32_t *result) { return emit(1, result); }
''')
    subprocess.run(["gcc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    ffi = FFI()
    ffi.cdef("int transform(int32_t *result); int emit(int tag, int32_t *result); "
             "int emit_utf8(int32_t *result); int emit_invalid(int32_t *result);")
    native = ffi.dlopen(str(library))
    module = SimpleNamespace(ffi=ffi, lib=SimpleNamespace(transform=native.transform))
    monkeypatch.setattr(compiler, "build_compiled_module", lambda definition: module)
    header = generate_header(Signature.from_dict(yaml.safe_load(SCHEMA)))
    values = {"__schema__": SCHEMA, "__header__": header, "__compilation__": {}}
    monkeypatch.setattr(run, "_resolve_dunder_value", lambda tf, dunder, key, celltype: values[key])

    def call():
        return run.call_compiled_transform({"__language__": "c"}, {}, "unused", {}, "plain", {})

    return call, module.lib, native


@pytest.fixture
def routed(monkeypatch):
    from contextlib import contextmanager
    # The actual worker initializer sets the marker read by this predicate.
    monkeypatch.setattr(run, "is_worker", lambda: True)

    @contextmanager
    def during_call():
        # Pytest replaces its capture streams between fixture setup and call.
        previous = sys.stdout, sys.stderr, threading.Thread.start
        install_output_routing()
        try:
            yield
        finally:
            sys.stdout, sys.stderr, threading.Thread.start = previous

    return during_call


def test_native_printf_and_fprintf_are_flushed_into_separate_request_streams(compiled_call, routed):
    with routed():
        call, _, _ = compiled_call
        stdout, stderr = io.StringIO(), io.StringIO()
        with output_route(stdout, stderr):
            assert call() == 1
        assert stdout.getvalue() == "native-out-1"
        assert stderr.getvalue() == "native-err-1"


def test_overlapping_compiled_calls_serialize_and_keep_request_routes(compiled_call, routed):
    with routed():
        call, lib, native = compiled_call
        entered, release, attempting, second_entered = (threading.Event() for _ in range(4))
        outputs = [(io.StringIO(), io.StringIO()) for _ in range(2)]
        errors = []

        def transform(result):
            tag = int(threading.current_thread().name[-1])
            if tag == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("first compiled call timed out")
            else:
                second_entered.set()
            return native.emit(tag, result)

        lib.transform = transform

        def execute(index):
            try:
                with output_route(*outputs[index]):
                    if index == 1:
                        attempting.set()
                    assert call() == index + 1
            except BaseException as exc:
                errors.append(exc)

        first = threading.Thread(target=execute, args=(0,), name="compiled-1")
        second = threading.Thread(target=execute, args=(1,), name="compiled-2")
        first.start()
        try:
            assert entered.wait(5)
            second.start()
            assert attempting.wait(5)
            assert not second_entered.wait(0.15), "compiled calls overlapped in one child"
        finally:
            release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        assert not first.is_alive() and not second.is_alive()
        assert not errors
        assert second_entered.is_set()
        for tag, (stdout, stderr) in enumerate(outputs, 1):
            assert stdout.getvalue() == f"native-out-{tag}"
            assert stderr.getvalue() == f"native-err-{tag}"


def test_process_compiled_output_stays_on_caller_fds(compiled_call, routed, monkeypatch, capfd):
    with routed():
        monkeypatch.setattr(run, "is_worker", lambda: False)
        call, _, _ = compiled_call
        stdout, stderr = io.StringIO(), io.StringIO()
        with output_route(stdout, stderr):
            assert call() == 1
        ctypes.CDLL(None).fflush(None)
        captured = capfd.readouterr()
        assert captured.out == "native-out-1"
        assert captured.err == "native-err-1"
        assert stdout.getvalue() == stderr.getvalue() == ""


def test_compiled_exception_flushes_output_restores_fds_and_releases_lock(compiled_call, routed, capfd):
    with routed():
        call, lib, native = compiled_call
        stdout, stderr = io.StringIO(), io.StringIO()

        def fail(result):
            native.emit(3, result)
            raise ValueError("native wrapper failure")

        lib.transform = fail
        with output_route(stdout, stderr):
            with pytest.raises(RuntimeError, match="native wrapper failure"):
                call()
        assert stdout.getvalue() == "native-out-3"
        assert stderr.getvalue() == "native-err-3"
        import os
        os.write(1, b"restored-out")
        os.write(2, b"restored-err")
        restored = capfd.readouterr()
        assert restored.out == "restored-out"
        assert restored.err == "restored-err"
        lib.transform = native.transform
        completed = threading.Event()
        errors = []

        def again():
            try:
                with output_route(io.StringIO(), io.StringIO()):
                    assert call() == 1
            except BaseException as exc:
                errors.append(exc)
            finally:
                completed.set()

        thread = threading.Thread(target=again, daemon=True)
        thread.start()
        assert completed.wait(5), "compiled capture lock leaked after exception"
        thread.join(5)
        assert not errors


def test_native_stream_delivery_waits_for_compiled_call_return(compiled_call, routed):
    with routed():
        call, lib, native = compiled_call
        emitted, release = threading.Event(), threading.Event()
        stdout, stderr = io.StringIO(), io.StringIO()
        errors = []

        def blocked(result):
            status = native.emit(4, result)
            ctypes.CDLL(None).fflush(None)
            emitted.set()
            if not release.wait(5):
                raise RuntimeError("compiled call timed out")
            return status

        lib.transform = blocked

        def execute():
            try:
                with output_route(stdout, stderr):
                    assert call() == 4
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=execute)
        thread.start()
        try:
            assert emitted.wait(5)
            assert stdout.getvalue() == stderr.getvalue() == ""
        finally:
            release.set()
            thread.join(5)
        assert not thread.is_alive()
        assert not errors
        assert stdout.getvalue() == "native-out-4"
        assert stderr.getvalue() == "native-err-4"


@pytest.mark.parametrize(
    "native_function, result, expected",
    [("emit_utf8", 5, "€" * 1000), ("emit_invalid", 6, "a\ufffdb")],
)
def test_native_output_decodes_complete_utf8_streams(
    compiled_call, routed, native_function, result, expected
):
    with routed():
        call, lib, native = compiled_call
        lib.transform = getattr(native, native_function)
        stdout, stderr = io.StringIO(), io.StringIO()
        with output_route(stdout, stderr):
            assert call() == result
        # A 3000-byte UTF-8 stream crosses wurlitzer's 1024-byte reads within
        # code points; decode the assembled stream, not individual chunks.
        assert stdout.getvalue() == expected
        assert stderr.getvalue() == expected
