"""Cross-backend deep-pin contract coverage; Python memory cases start Phase 3a."""

import hashlib

import numpy as np
import pytest

from seamless import Buffer, CacheMissError, Checksum
from seamless_transformer import Transformer
from seamless_transformer.run import run_transformation_dict
from seamless_transformer.transformation_utils import extract_tf_dunder


def _held(value, celltype):
    buffer = Buffer(value, celltype)
    buffer.tempref()
    return buffer


def _python_transformer(code, value, celltype, output="mixed", pin="data"):
    transformer = Transformer("python")
    transformer.code = code
    transformer.celltypes[pin] = celltype
    transformer.pins[pin] = value.get_checksum() if isinstance(value, Buffer) else value
    transformer.celltypes.result = output
    transformer.local = True
    return transformer


LANGUAGE_BACKENDS = [
    ("python", "mem"),
    ("python", "cli"),
    ("r", "mem"),
    ("r", "cli"),
    ("julia", "mem"),
    ("julia", "cli"),
]


def _transformer(language, backend, code, value, celltype, output="mixed"):
    transformer = Transformer(language)
    if language in ("python", "r", "julia"):
        transformer.metavars.BACKEND = backend
    transformer.code = code
    transformer.celltypes.value = celltype
    if isinstance(value, Buffer):
        transformer.pins.value = (
            value.get_value("checksum")
            if celltype == "checksum"
            else value.get_checksum()
        )
    else:
        transformer.pins.value = value
    transformer.celltypes.result = output
    transformer.local = True
    return transformer


def _identity_code(language):
    return {
        "python": "result = value",
        "r": "result <- value",
        "julia": "result = value",
    }[language]


def _filter_keep_code(language):
    return {
        "python": "result = {key: item for key, item in value.items() if key == 'keep'}",
        "r": "result <- value[names(value) == 'keep']",
        "julia": 'result = Dict(k => v for (k, v) in value if k == "keep")',
    }[language]


def _run_checksum(transformer):
    handle = transformer.build()
    checksum = handle.construct()
    assert checksum is not None, handle.exception
    payload = checksum.resolve().get_value("plain")
    payload.update(extract_tf_dunder(handle._definition_payload_template))
    payload.update(handle._tf_dunder)
    result = Checksum(run_transformation_dict(payload, checksum, scratch=True, require_value=True))
    if payload.get("__language__") == "python" and "__env__" not in payload:
        # Phase 5: identical parsed deep pins and result classification under
        # the other Python backend; both executions bypass the result cache.
        other = dict(payload)
        meta = dict(other.get("__meta__", {}))
        metavars = dict(meta.get("metavars", {}))
        metavars["BACKEND"] = "mem" if metavars.get("BACKEND", "mem") == "cli" else "cli"
        meta["metavars"] = metavars
        other["__meta__"] = meta
        assert Checksum(run_transformation_dict(other, checksum, scratch=True, require_value=True)) == result
    return result


def _run_value(transformer):
    result = _run_checksum(transformer).resolve()
    celltype = transformer.celltypes.result
    return result.content if celltype == "bytes" else result.get_value(celltype)


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_folder_pin_is_contents_including_read_buffer_index(language, backend):
    member = _held(b"folder bytes", "bytes")
    index = _held({"member.txt": member.get_checksum().hex()}, "plain")
    result = _run_value(
        _transformer(
            language,
            backend,
            _identity_code(language),
            index,
            "folder",
        )
    )
    assert result == {"member.txt": b"folder bytes"}


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder"])
def test_deep_pins_are_checksum_objects_without_resolution(
    language, backend, celltype, monkeypatch
):
    checksum = "a" * 64
    index = _held({"missing": checksum}, "plain")

    original_resolve = Checksum.resolve

    def no_resolve(self):
        if self.hex() == checksum:
            raise AssertionError("deep pins carry references without resolution")
        return original_resolve(self)

    monkeypatch.setattr(Checksum, "resolve", no_resolve)
    code = {
        "python": (
            "result = {key: [type(item).__name__, item.hex()] "
            "for key, item in value.items()}"
        ),
        "r": "result <- list(class = class(value$missing)[1], hex = as.character(value$missing))",
        "julia": 'result = Dict("class" => string(typeof(value["missing"])), "hex" => value["missing"].hex)',
    }[language]
    result = _run_value(
        _transformer(language, backend, code, index, celltype, output="plain")
    )
    if language == "python":
        assert result == {"missing": ["Checksum", checksum]}
    else:
        assert result["class"].endswith("Checksum") or result["class"] == "seamless_checksum"
        assert result["hex"] == checksum


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_filtered_deepcell_result_has_filtered_index_checksum(language, backend):
    keep = Buffer({"value": 1}, "mixed").get_checksum()
    drop = Buffer({"value": 2}, "mixed").get_checksum()
    index = _held({"keep": keep.hex(), "drop": drop.hex()}, "plain")
    result = _run_checksum(
        _transformer(
            language,
            backend,
            _filter_keep_code(language),
            index,
            "deepcell",
            output="deepcell",
        )
    )
    expected = Buffer({"keep": keep.hex()}, "deepcell").get_checksum()
    assert result == expected


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_deepfolder_source_plain_and_folder_casts(language, backend):
    member = _held(b"cast source", "bytes")
    index = _held({"member": member.get_checksum().hex()}, "plain")
    plain_result = _run_value(
        _transformer(
            language,
            backend,
            _identity_code(language),
            index,
            "plain",
            output="mixed",
        )
    )
    folder_result = _run_value(
        _transformer(
            language,
            backend,
            _identity_code(language),
            index,
            "folder",
            output="mixed",
        )
    )
    assert plain_result == {"member": member.get_checksum().hex()}
    assert folder_result == {"member": b"cast source"}


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_deep_results_mix_values_and_checksum_objects(language, backend):
    referenced = Checksum("b" * 64)
    code = {
        "python": (
            "from seamless import Checksum\n"
            "result = {'value': b'raw bytes', 'reference': Checksum('" + "b" * 64 + "')}"
        ),
        "r": (
            "result <- list(value = charToRaw('raw bytes'), "
            "reference = seamless_checksum('" + "b" * 64 + "'))"
        ),
        "julia": (
            'result = Dict("value" => Vector{UInt8}(codeunits("raw bytes")), '
            '"reference" => Checksum("' + "b" * 64 + '"))'
        ),
    }[language]
    transformer = _transformer(language, backend, code, {}, "plain", output="deepcell")
    result = _run_checksum(transformer).resolve().get_value("deepcell")
    assert result == {
        "value": Buffer(b"raw bytes", "mixed").get_checksum().hex(),
        "reference": referenced.hex(),
    }


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_hex_string_members_are_values_unless_wrapped(language, backend):
    hex_value = "c" * 64
    code = {
        "python": (
            "from seamless import Checksum\n"
            "result = {'value': '" + hex_value + "', 'checksum': Checksum('" + hex_value + "')}"
        ),
        "r": (
            "result <- list(value = '" + hex_value + "', checksum = seamless_checksum('"
            + hex_value + "'))"
        ),
        "julia": (
            'result = Dict("value" => "' + hex_value + '", "checksum" => Checksum("'
            + hex_value + '"))'
        ),
    }[language]
    transformer = _transformer(language, backend, code, {}, "plain", output="deepcell")
    result = _run_checksum(transformer).resolve().get_value("deepcell")
    assert result["value"] == Buffer(hex_value, "mixed").get_checksum().hex()
    assert result["checksum"] == hex_value

    folder_code = {
        "python": "result = {'value': '" + hex_value + "'}",
        "r": "result <- list(value = '" + hex_value + "')",
        "julia": 'result = Dict("value" => "' + hex_value + '")',
    }[language]
    with pytest.raises(Exception):
        _run_checksum(
            _transformer(language, backend, folder_code, {}, "plain", output="folder")
        )


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_plain_index_returned_unwrapped_produces_string_members(language, backend):
    checksum = "d" * 64
    index = _held({"member": checksum}, "plain")
    result = _run_checksum(
        _transformer(
            language,
            backend,
            _identity_code(language),
            index,
            "plain",
            output="deepcell",
        )
    ).resolve().get_value("deepcell")
    assert result["member"] == Buffer(checksum, "mixed").get_checksum().hex()


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_deepcell_complex_members_checksum_parity(language, backend):
    matrix = np.array([[1, 2], [3, 4]], dtype=np.int64)
    magic_bytes = b"\x93NUMPY raw member bytes"
    result_value = {
        "array": matrix,
        "bytes": magic_bytes,
        "small": {"one": 1, "two": [2, 3]},
    }
    code = {
        "python": (
            "import numpy as np\n"
            "result = {'array': np.array([[1, 2], [3, 4]], dtype=np.int64), "
            "'bytes': b'\\x93NUMPY raw member bytes', "
            "'small': {'one': 1, 'two': [2, 3]}}"
        ),
        "r": (
            "result <- list(array = matrix(as.integer(c(1, 2, 3, 4)), nrow=2, byrow=TRUE), "
            "bytes = as.raw(c(147, 78, 85, 77, 80, 89, 32, 114, 97, 119, 32, 109, 101, "
            "109, 98, 101, 114, 32, 98, 121, 116, 101, 115)), "
            "small = list(one = 1, two = c(2, 3)))"
        ),
        "julia": (
            'result = Dict("array" => [1 2; 3 4], '
            '"bytes" => UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x20, 0x72, 0x61, '
            '0x77, 0x20, 0x6d, 0x65, 0x6d, 0x62, 0x65, 0x72, 0x20, 0x62, 0x79, 0x74, '
            '0x65, 0x73], "small" => Dict("one" => 1, "two" => [2, 3]))'
        ),
    }[language]
    transformer = _transformer(language, backend, code, {}, "plain", output="deepcell")
    result = _run_checksum(transformer).resolve().get_value("deepcell")
    expected = {
        key: Buffer(value, "mixed").get_checksum().hex()
        for key, value in result_value.items()
    }
    actual_members = {
        key: Checksum(checksum).resolve().get_value("mixed")
        for key, checksum in result.items()
    }
    assert result == expected, actual_members


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_deep_result_member_keys_are_preserved(language, backend):
    code = {
        "python": "result = {'json': 1, '__checksum__': 2, 'nested/key': 3}",
        "r": 'result <- setNames(list(1, 2, 3), c("json", "__checksum__", "nested/key"))',
        "julia": 'result = Dict("json" => 1, "__checksum__" => 2, "nested/key" => 3)',
    }[language]
    transformer = _transformer(language, backend, code, {}, "plain", output="deepcell")
    result = _run_checksum(transformer).resolve().get_value("deepcell")
    assert set(result) == {"json", "__checksum__", "nested/key"}


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_hashing_folder_contents_checksum_parity(language, backend):
    member = _held(b"hash this folder member", "bytes")
    index = _held({"member": member.get_checksum().hex()}, "plain")
    code = {
        "python": "import hashlib\nresult = hashlib.sha256(value['member']).hexdigest()",
        "r": "writeBin(value$member, p <- tempfile()); result <- strsplit(system2('sha256sum', p, stdout=TRUE), ' ')[[1]][1]",
        "julia": 'using SHA\nresult = bytes2hex(sha256(value["member"]))',
    }[language]
    transformer = _transformer(
        language,
        backend,
        code,
        index,
        "folder",
        output="plain",
    )
    assert _run_value(transformer) == hashlib.sha256(b"hash this folder member").hexdigest()


@pytest.mark.parametrize("language", ["r", "julia", "python"])
@pytest.mark.parametrize("runtime", ["docker", "podman", "apptainer"])
@pytest.mark.parametrize("celltype", ["folder", "deepfolder"])
def test_container_folder_and_deepfolder_pin_forms(
    language, runtime, celltype, language_runtime_images, monkeypatch, tmp_path
):
    images = language_runtime_images
    monkeypatch.setenv("SEAMLESS_CONTAINER_RUNTIME", runtime)
    monkeypatch.setenv("SEAMLESS_CONTAINER_PULL", "never")
    monkeypatch.delenv("SEAMLESS_CONTAINER_ARGS", raising=False)
    if runtime == "apptainer":
        monkeypatch.setenv("SEAMLESS_APPTAINER_DIR", str(images["sif_dir"]))
    else:
        monkeypatch.delenv("SEAMLESS_APPTAINER_DIR", raising=False)

    external = tmp_path / "outside-buffer-file"
    external.write_bytes(b"container external symlink member")
    read_dir = tmp_path / "read-buffer"
    read_dir.mkdir()
    import seamless_transformer.transformation_namespace as namespace

    monkeypatch.setattr(namespace, "_get_read_folder_directories", lambda: [str(read_dir)])
    member_buffer = _held(external.read_bytes(), "bytes")
    checksum = member_buffer.get_checksum().hex()
    stored = read_dir / checksum[:2] / checksum
    stored.parent.mkdir()
    stored.symlink_to(external)

    if language == "python":
        index = _held({"member": checksum}, "plain")
        if celltype == "deepfolder":
            code = f"runtime_variant = {runtime!r}\nresult = value[\"member\"].hex()"
            output = "plain"
        else:
            code = f"runtime_variant = {runtime!r}\nresult = value[\"member\"]"
            output = "bytes"
        transformer = _transformer(
            language, "cli", code, index, celltype, output=output
        )
        transformer.environment.set_docker({"name": images[language]})
        if celltype == "deepfolder":
            assert _run_value(transformer) == checksum
        else:
            assert _run_value(transformer) == external.read_bytes()
        return

    from seamless_transformer.languages import get_language_handler

    if celltype == "deepfolder":
        code = (
            'result <- as.character(data[["member"]])'
            if language == "r"
            else 'result = data["member"].hex'
        )
        output_celltype = "plain"
        member_value = member_buffer.get_checksum()
    else:
        code = (
            'result <- data[["member"]]'
            if language == "r"
            else 'result = data["member"]'
        )
        output_celltype = "bytes"
        member_value = Buffer(external.read_bytes(), "bytes")
    handler = get_language_handler(language).handler
    arguments = dict(
        code=code,
        pins={
            "data": {
                "member": (
                    member_value
                )
            }
        },
        filesystem={"data": {"celltype": celltype, "filesystem": False}},
        output_name="result",
        output_celltype=output_celltype,
        metavars={"BACKEND": "cli"},
        environment={"docker": {"name": images[language]}},
        process_lock=True,
    )

    expected = checksum if celltype == "deepfolder" else external.read_bytes()
    assert handler(**arguments) == expected


def test_compiled_transformer_still_refuses_deep_pin():
    import yaml

    transformer = Transformer("c", compiled=True)
    transformer.schema = yaml.safe_dump(
        {
            "inputs": [{"name": "x", "dtype": "int32"}],
            "outputs": [{"name": "result", "dtype": "int32"}],
        }
    )
    with pytest.raises(Exception):
        transformer.celltypes.x = "deepcell"


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_checksum_pin_is_runtime_checksum_object(language, backend):
    checksum = Checksum("e" * 64)
    code = {
        "python": "result = [type(value).__name__, value.hex()]",
        "r": "result <- list(class = class(value)[1], hex = as.character(value))",
        "julia": 'result = Dict("class" => string(typeof(value)), "hex" => value.hex)',
    }[language]
    result = _run_value(
        _transformer(
            language,
            backend,
            code,
            Buffer(checksum, "checksum"),
            "checksum",
            output="plain",
        )
    )
    if language == "python":
        assert result == ["Checksum", checksum.hex()]
    else:
        assert result["class"].endswith("Checksum") or result["class"] == "seamless_checksum"
        assert result["hex"] == checksum.hex()


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_checksum_object_result_celltype_restrictions(language, backend):
    checksum = Checksum("e" * 64)
    make_code = {
        "python": "from seamless import Checksum\nresult = Checksum('" + checksum.hex() + "')",
        "r": "result <- seamless_checksum('" + checksum.hex() + "')",
        "julia": 'result = Checksum("' + checksum.hex() + '")',
    }[language]
    assert _run_value(
        _transformer(language, backend, make_code, {}, "plain", output="checksum")
    ) == checksum

    string_code = {
        "python": "result = '" + checksum.hex() + "'",
        "r": "result <- '" + checksum.hex() + "'",
        "julia": 'result = "' + checksum.hex() + '"',
    }[language]
    assert _run_value(
        _transformer(language, backend, string_code, {}, "plain", output="checksum")
    ) == checksum

    reject_code = {
        "python": "from seamless import Checksum\nresult = {'checksum': Checksum('" + checksum.hex() + "')}",
        "r": "result <- list(checksum = seamless_checksum('" + checksum.hex() + "'))",
        "julia": 'result = Dict("checksum" => Checksum("' + checksum.hex() + '"))',
    }[language]
    with pytest.raises(Exception):
        _run_value(_transformer(language, backend, reject_code, {}, "plain", output="mixed"))


@pytest.mark.parametrize(("language", "backend"), LANGUAGE_BACKENDS)
def test_handler_deep_result_rejects_nonstring_keys_and_unsupported_values(language, backend):
    code = {
        "python": "result = {1: 'not a string key'}",
        "r": "result <- list(unsupported = new.env())",
        "julia": 'result = Dict(1 => "not a string key")',
    }[language]
    transformer = _transformer(language, backend, code, {}, "plain", output="deepcell")
    with pytest.raises(Exception):
        _run_checksum(transformer)
