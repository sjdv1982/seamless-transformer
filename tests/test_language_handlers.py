"""Core handler-language contract, exercised without R or Julia."""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import pytest

from seamless import Buffer, Checksum
from seamless_transformer import Transformer
from seamless_transformer.languages import (
    available_languages, get_language_handler, is_handler_language,
    register_language_handler,
)
from seamless_transformer.run import run_transformation_dict
from seamless_transformer.transformation_utils import tf_get_buffer


def toy_handler(code, pins, filesystem, output_name, output_celltype,
                metavars, environment, process_lock):
    import os
    return {
        "code": code, "pin_names": sorted(pins),
        "output_name": output_name, "output_celltype": output_celltype,
        "metavars": metavars, "environment": environment,
        "process_lock": process_lock, "pid": os.getpid(),
        "leaked": sorted(set(globals()) & {"value", "PINS", "FILESYSTEM", "OUTPUTPIN"}),
    }


def alternate_handler(pins):
    return {"alternate": True, "pin_names": sorted(pins)}


def invalid_handler(not_offered):
    return not_offered


def pin_handler(pins, filesystem):
    value = pins["value"]
    if filesystem["value"]["filesystem"]:
        return {"on_file": True, "path": value, "celltype": filesystem["value"]["celltype"]}
    return {"on_file": False, "celltype": filesystem["value"]["celltype"],
            "value": value.tolist() if hasattr(value, "tolist") else value.hex() if isinstance(value, bytes) else value}


@pytest.fixture
def language():
    name = "toy-" + uuid.uuid4().hex
    register_language_handler(name, toy_handler, metavars={"BACKEND": "cli", "OTHER": 7})
    return name


def builder(language, celltype="plain", value=12):
    tf = Transformer(language)
    tf.code = "arbitrary foreign source"
    tf.celltypes.result = "plain"
    tf.celltypes.value = celltype
    tf.pins.value = value
    tf.local = True
    return tf


def execution_dict(tf):
    from seamless_transformer.transformation_utils import extract_tf_dunder
    handle = tf.build()
    checksum = handle.construct()
    assert checksum is not None, handle.exception
    payload = checksum.resolve().get_value("plain")
    # Construction retains orthogonal fields in the definition envelope.
    payload.update(extract_tf_dunder(handle._definition_payload_template))
    payload.update(handle._tf_dunder)
    return handle, checksum, payload


def execute(payload, checksum):
    result = run_transformation_dict(payload, checksum)
    return Checksum(result).resolve().get_value("plain")


def test_registration_and_available_language_kinds(language):
    assert is_handler_language(language)
    assert get_language_handler(language) is not None
    assert not is_handler_language("python")
    listing = available_languages()
    assert listing["python"] == listing["bash"] == "native"
    assert listing["c"] == "compiled"
    assert listing[language] == "handler"
    assert "ipython" not in listing


@pytest.mark.parametrize("name", ["python", "bash", "c"])
def test_registration_rejects_cross_kind_collision(name):
    with pytest.raises((ValueError, TypeError)):
        register_language_handler(name, toy_handler)


def test_registration_rejects_unoffered_parameter():
    with pytest.raises((ValueError, TypeError)):
        register_language_handler("invalid-" + uuid.uuid4().hex, invalid_handler)


@pytest.mark.parametrize("direct,class_name", [(False, "HandlerTransformer"), (True, "DirectHandlerTransformer")])
def test_factory_handler_classes_and_read_only_language(language, direct, class_name):
    tf = Transformer(language, direct=direct)
    assert type(tf).__name__ == class_name
    assert tf.code is None
    assert tf.language == language
    with pytest.raises(AttributeError):
        tf.language = "python"


def test_metavar_defaults_survive_partial_override_and_reject_unknown(language):
    tf = builder(language)
    assert tf.metavars.BACKEND == "cli"
    tf.metavars.BACKEND = "memory"
    with pytest.raises((AttributeError, KeyError, ValueError)):
        tf.metavars.UNDECLARED = 1
    handle, checksum, payload = execution_dict(tf)
    assert handle.meta["metavars"] == {"BACKEND": "memory", "OTHER": 7}
    result = execute(payload, checksum)
    assert result["metavars"] == {"BACKEND": "memory", "OTHER": 7}


def test_handler_source_and_backend_are_orthogonal_but_celltype_is_not(language):
    tf = builder(language)
    first, first_checksum, first_dict = execution_dict(tf)
    first_source = first_dict["__handler__"]
    assert Checksum(first_source).resolve().get_value("text").startswith("def toy_handler(")
    tf.metavars.BACKEND = "memory"
    second, second_checksum, second_dict = execution_dict(tf)
    assert second_checksum == first_checksum
    register_language_handler(language, alternate_handler, metavars={"BACKEND": "cli", "OTHER": 7})
    third, third_checksum, third_dict = execution_dict(tf)
    assert third_dict["__handler__"] != first_source
    assert third_checksum == first_checksum
    assert tf_get_buffer(first_dict).get_checksum() == tf_get_buffer(third_dict).get_checksum()
    tf.celltypes.value = "mixed"
    fourth = tf.build()
    assert fourth.construct() != first_checksum
    # Retain all snapshots until assertions finish, including their source refs.
    assert all(item is not None for item in (first, second, third))


@pytest.mark.parametrize("celltype,value", [
    ("plain", {"x": 3}), ("mixed", [1, 2]), ("text", "hello"),
    ("bytes", b"\x00\xff"), ("binary", np.array([1, 2], dtype=np.int32)),
])
def test_neutral_filesystem_path_and_value_forms(language, celltype, value, tmp_path, monkeypatch):
    register_language_handler(language, pin_handler)
    tf = builder(language, celltype, value)
    handle, checksum, payload = execution_dict(tf)
    assert payload["__format__"]["value"]["filesystem"] == {"mode": "file", "optional": True}
    namespace_module = importlib.import_module("seamless_transformer.transformation_namespace")
    buffer = Buffer(value, celltype)
    path = tmp_path / buffer.get_checksum().hex()
    path.write_bytes(buffer.content)
    monkeypatch.setattr(namespace_module, "_get_read_folder_directories", lambda: [str(tmp_path)])
    result = execute(payload, checksum)
    assert result == {"on_file": True, "path": str(path), "celltype": celltype}
    monkeypatch.setattr(namespace_module, "_get_read_folder_directories", lambda: [])
    result = execute(payload, checksum)
    expected = value.tolist() if hasattr(value, "tolist") else value.hex() if isinstance(value, bytes) else value
    assert result == {"on_file": False, "celltype": celltype, "value": expected}
    assert handle is not None


def test_registry_fallback_offered_arguments_and_fresh_namespace(language):
    tf = builder(language)
    handle, checksum, payload = execution_dict(tf)
    payload.pop("__handler__")
    payload["__output__"] = ("answer", "plain", None)
    environment_buffer = Buffer({"custom": "environment"}, "plain")
    environment_buffer.tempref()
    environment_buffer.transfer_write()
    payload["__env__"] = environment_buffer.get_checksum().hex()
    result = execute(payload, checksum)
    assert result["code"].rstrip("\n") == "arbitrary foreign source"
    assert result["pin_names"] == ["value"]
    assert result["output_name"] == "answer"
    assert result["output_celltype"] == "plain"
    assert result["environment"] == {"custom": "environment"}
    assert result["leaked"] == []
    assert result["process_lock"] is True
    assert handle is not None


def test_handler_receives_process_lock_optout(language):
    tf = builder(language)
    tf.process_lock = False
    handle, checksum, payload = execution_dict(tf)
    assert execute(payload, checksum)["process_lock"] is False
    assert handle is not None


def _driver_only_handler(mode):
    import seamless
    from seamless_transformer import spawn
    name = "driver-only-" + uuid.uuid4().hex
    if mode == "spawn":
        spawn(1)
    try:
        register_language_handler(name, toy_handler, metavars={"BACKEND": "cli"})
        tf = builder(name)
        tf.local = mode != "spawn"
        result = tf.build().run()
        assert result["pin_names"] == ["value"]
        assert result["process_lock"] is True
        assert (result["pid"] != os.getpid()) == (mode == "spawn")
    finally:
        seamless.close()


@pytest.mark.parametrize("mode", ["inprocess", "spawn"])
def test_handler_registered_only_in_driver_executes(mode):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    script = f"import test_language_handlers as m; m._driver_only_handler({mode!r})"
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(autouse=True)
def isolated_language_registries(monkeypatch):
    module = importlib.import_module("seamless_transformer.languages")
    monkeypatch.setattr(module, "_registry", dict(module._registry))
    monkeypatch.setattr(module, "_handler_registry", dict(module._handler_registry))


def pins(pins):
    return pins["value"] + 1


def test_handler_function_name_can_equal_an_offered_argument(language):
    register_language_handler(language, pins)
    tf = builder(language)
    handle, checksum, payload = execution_dict(tf)
    assert execute(payload, checksum) == 13
    assert handle is not None


@pytest.mark.parametrize("name_kind", ["python", "bash", "handler"])
def test_compiled_registration_rejects_native_and_handler_names(language, name_kind):
    from dataclasses import asdict
    from seamless_transformer.languages import define_compiled_language, get_language
    name = language if name_kind == "handler" else name_kind
    previous = available_languages()[name]
    with pytest.raises(ValueError):
        define_compiled_language(name, asdict(get_language("c").compilation))
    assert available_languages()[name] == previous


def test_compiled_registration_can_replace_its_own_kind():
    from dataclasses import asdict
    from seamless_transformer.languages import define_compiled_language, get_language
    name = "compiled-toy-" + uuid.uuid4().hex
    config = asdict(get_language("c").compilation)
    define_compiled_language(name, config)
    config["options"] = ["-O0"]
    define_compiled_language(name, config)
    assert available_languages()[name] == "compiled"
    assert get_language(name).compilation.options == ["-O0"]
