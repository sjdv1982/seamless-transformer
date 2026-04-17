import shutil

import pytest

from seamless_transformer import DirectCompiledTransformer

pytestmark = pytest.mark.skipif(not shutil.which("go"), reason="go required")


ADD_SCHEMA = """\
inputs:
  - {name: a, dtype: int32}
  - {name: b, dtype: int32}
outputs:
  - {name: result, dtype: int32}
"""

# Go uses CGO to export a C-callable transform().
# The preamble #include <stdint.h> makes C.int32_t available.
# The schema-generated C header declares:
#   int transform(int32_t a, int32_t b, int32_t *result);
# which matches the //export signature below.
ADD_GO = """\
package main

/*
#include <stdint.h>
*/
import "C"

//export transform
func transform(a C.int32_t, b C.int32_t, result *C.int32_t) C.int {
  *result = a + b
  return 0
}
func main() {}
"""


def test_go_scalar():
    tf = DirectCompiledTransformer("go")
    tf.schema = ADD_SCHEMA
    tf.code = ADD_GO
    assert tf(a=13, b=16) == 29


def test_go_scalar_larger_values():
    tf = DirectCompiledTransformer("go")
    tf.schema = ADD_SCHEMA
    tf.code = ADD_GO
    assert tf(a=80, b=100) == 180
