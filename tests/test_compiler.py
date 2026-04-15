import shutil

import pytest

from seamless_transformer.compiler import build_compiled_module
from seamless_transformer.compiler.compile import _merge_objects, compile, complete
from seamless_transformer.compiler.cffi_wrapper import build_extension_cffi


pytestmark = pytest.mark.skipif(not shutil.which("gcc"), reason="gcc required")


HEADER = """\
#include <stdint.h>
int transform(
    int32_t a,
    int32_t b,
    int32_t *result
);
"""


SOURCE = """\
#include <stdint.h>
int transform(int32_t a, int32_t b, int32_t *result) {
    *result = a + b;
    return 0;
}
"""


def module_definition():
    return {
        "type": "compiled",
        "target": "profile",
        "link_options": [],
        "public_header": {"language": "c", "code": HEADER},
        "objects": {"main": {"language": "c", "code": SOURCE}},
    }


def test_complete_and_compile_object():
    completed = complete(module_definition())
    assert completed["objects"]["main"]["compiler_binary"] == "gcc"
    objects = compile(completed)
    assert list(objects) == ["main.o"]
    assert objects["main.o"]
    assert _merge_objects(objects, "object") == objects
    assert _merge_objects({"lib.a": b"archive"}, "archive") == b"archive"


def test_build_extension_cffi():
    completed = complete(module_definition())
    objects = compile(completed)
    so = build_extension_cffi(
        "_test_build_extension_cffi",
        objects,
        "profile",
        HEADER,
        [],
    )
    assert so.startswith(b"\x7fELF") or so


def test_build_compiled_module():
    module = build_compiled_module(module_definition(), module_name="_test_compiled_add")
    result = module.ffi.new("int32_t *")
    status = module.lib.transform(2, 3, result)
    assert status == 0
    assert result[0] == 5
