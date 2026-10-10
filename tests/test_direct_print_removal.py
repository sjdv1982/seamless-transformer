"""Phase B acceptance: retired direct_print and CLI streaming isolation."""

import dataclasses
import importlib
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from seamless import Checksum
from seamless_transformer import Transformer
from seamless_transformer.compiled_transformer import CompiledTransformer
from seamless_transformer.frozen_transformer import FrozenTransformer
from seamless_transformer.transformer_class import BashTransformer, PythonTransformer


@pytest.mark.parametrize("language,compiled", [("python", False), ("bash", False), ("c", True)])
def test_direct_print_attribute_is_rejected(language, compiled):
    tf = Transformer(language, compiled=compiled)
    with pytest.raises(AttributeError):
        getattr(tf, "direct_print")
    with pytest.raises(AttributeError):
        tf.direct_print = True
    assert "__direct_print__" not in tf.meta


@pytest.mark.parametrize("constructor,args", [(PythonTransformer, ()), (BashTransformer, ()), (CompiledTransformer, ("c",))])
def test_direct_print_constructor_keyword_is_rejected(constructor, args):
    with pytest.raises(TypeError, match="direct_print"):
        constructor(*args, direct_print=True)


def test_frozen_transformer_has_no_direct_print_field():
    assert "direct_print" not in {field.name for field in dataclasses.fields(FrozenTransformer)}


@pytest.mark.parametrize("module,args", [
    ("seamless_transformer.cmd.api.main", ["--direct-print", "-c", "echo hello"]),
    ("seamless_transformer.api.run_transformation", ["--direct-print", "1" * 64]),
])
def test_cli_rejects_direct_print(module, args, capsys):
    with pytest.raises(SystemExit) as exc:
        importlib.import_module(module)._main(args)
    assert exc.value.code == 2
    assert "--direct-print" in capsys.readouterr().err


def _emit_stream_text():
    from seamless_dask.client import SeamlessDaskClient

    client = SeamlessDaskClient.__new__(SeamlessDaskClient)
    client._stream_topic_lock = threading.RLock()
    client._stream_topics = {}
    for stream in ("stdout", "stderr"):
        client._render_stream_event("abcdef1234567890", (0, {
            "kind": "stream", "stream": stream, "text": f"STREAM-{stream}\n",
        }))


def test_run_transformation_streaming_sets_flag_and_keeps_checksum_stdout(monkeypatch, capsys):
    from seamless_transformer.api import run_transformation as cli

    tf = SimpleNamespace(streaming=False)
    monkeypatch.setattr(cli.seamless_config, "init", lambda **kwargs: None)
    monkeypatch.setattr(cli, "_parse_checksum", lambda arg: Checksum("1" * 64))
    monkeypatch.setattr(cli, "_resolve_transformation_dict", lambda checksum: {"__output__": ("result", "mixed", None)})
    monkeypatch.setattr(cli, "_load_dunder", lambda *args: {})
    monkeypatch.setattr(cli, "transformation_from_dict", lambda *args, **kwargs: tf)

    def compute(transformation, **kwargs):
        assert transformation.streaming is True
        _emit_stream_text()
        return Checksum("2" * 64)

    monkeypatch.setattr(cli, "compute_transformation_sync", compute)
    assert cli._main(["--streaming", "1" * 64]) == 0
    output = capsys.readouterr()
    assert output.out == "2" * 64 + "\n"
    assert "STREAM-stdout" in output.err
    assert "STREAM-stderr" in output.err


def test_seamless_run_streaming_sets_flag_and_keeps_result_stdout(tmp_path):
    # A separate interpreter contains the CLI's global config/session lifetime.
    (tmp_path / "seamless.yaml").write_text("- execution: process\n")
    script = '''
import threading
from seamless import Checksum
from seamless_dask.client import SeamlessDaskClient
import seamless_transformer.transformation_class as tc
import seamless_transformer.cmd.get_results as results
from seamless_transformer.cmd.api.main import _main

def compute(tf, **kwargs):
    assert tf.streaming is True
    client = SeamlessDaskClient.__new__(SeamlessDaskClient)
    client._stream_topic_lock = threading.RLock()
    client._stream_topics = {}
    for stream in ("stdout", "stderr"):
        client._render_stream_event("abcdef1234567890", (0, {
            "kind": "stream", "stream": stream, "text": "STREAM-" + stream + "\\n",
        }))
    return Checksum("2" * 64)

tc.compute_transformation_sync = compute
results.get_result_buffer = lambda *args, **kwargs: None
results.get_results = lambda *args, **kwargs: "RESULT\\n"
status = _main(["--streaming", "-q", "-c", "echo hello"])
assert status in (None, 0), status
'''
    proc = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout == "RESULT\n"
    assert "STREAM-stdout" in proc.stderr
    assert "STREAM-stderr" in proc.stderr
