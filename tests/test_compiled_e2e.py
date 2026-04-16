import shutil

import numpy as np
import pytest

from seamless import Checksum
from seamless_transformer import (
    CompiledObject,
    CompiledTransformer,
    DirectCompiledTransformer,
)
from seamless_transformer.transformation_class import Transformation


pytestmark = pytest.mark.skipif(not shutil.which("gcc"), reason="gcc required")


ADD_SCHEMA = """\
inputs:
  - {name: a, dtype: int32}
  - {name: b, dtype: int32}
outputs:
  - {name: result, dtype: int32}
"""


ADD_C = """\
#include <stdint.h>
int transform(int32_t a, int32_t b, int32_t *result) {
    *result = a + b;
    return 0;
}
"""


def test_c_scalar_python_and_numpy_inputs():
    tf = DirectCompiledTransformer("c")
    tf.schema = ADD_SCHEMA
    tf.code = ADD_C
    assert tf(a=2, b=3) == 5
    assert tf(a=np.int32(4), b=np.int32(7)) == 11


@pytest.mark.skipif(not shutil.which("g++"), reason="g++ required")
def test_cpp_scalar():
    tf = DirectCompiledTransformer("cpp")
    tf.schema = ADD_SCHEMA
    tf.code = """\
#include <stdint.h>
extern "C" int transform(int32_t a, int32_t b, int32_t *result) {
    *result = a + b;
    return 0;
}
"""
    assert tf(a=5, b=6) == 11


@pytest.mark.skipif(not shutil.which("rustc"), reason="rustc required")
def test_rust_scalar_archive_mode():
    tf = DirectCompiledTransformer("rust")
    tf.schema = ADD_SCHEMA
    tf.code = """\
#[no_mangle]
pub unsafe extern "C" fn transform(a: i32, b: i32, result: *mut i32) -> i32 {
    unsafe {
        *result = a + b;
    }
    0
}
"""
    assert tf(a=6, b=7) == 13


def test_delayed_compiled_transformer():
    tf = CompiledTransformer("c")
    tf.schema = ADD_SCHEMA
    tf.code = ADD_C
    result = tf(a=2, b=9)
    assert isinstance(result, Transformation)
    assert result.run() == 11


def test_array_input_output_and_non_contiguous_normalization():
    tf = DirectCompiledTransformer("c")
    tf.schema = """\
inputs:
  - {name: arr, dtype: float64, shape: [N]}
  - {name: factor, dtype: float64}
outputs:
  - {name: result, dtype: float64, shape: [N]}
"""
    tf.code = """\
#include <stdint.h>
int transform(unsigned int N, const double *arr, double factor, double *result) {
    for (unsigned int i = 0; i < N; i++) result[i] = arr[i] * factor;
    return 0;
}
"""
    source = np.arange(8, dtype=np.float64)[::2]
    np.testing.assert_allclose(tf(arr=source, factor=2.5), source * 2.5)


def test_output_only_wildcard_slicing():
    tf = DirectCompiledTransformer("c")
    tf.schema = """\
inputs:
  - {name: arr, dtype: int32, shape: [N]}
outputs:
  - {name: values, dtype: int32, shape: [K]}
"""
    tf.metavars.maxK = 10
    tf.code = """\
#include <stdint.h>
int transform(unsigned int N, unsigned int maxK, const int32_t *arr, unsigned int *K, int32_t *values) {
    unsigned int k = 0;
    for (unsigned int i = 0; i < N && k < maxK; i++) {
        if (arr[i] > 0) values[k++] = arr[i];
    }
    *K = k;
    return 0;
}
"""
    np.testing.assert_array_equal(
        tf(arr=np.array([-1, 4, 0, 5], dtype=np.int32)),
        np.array([4, 5], dtype=np.int32),
    )


def test_structured_scalar_input_output():
    tf = DirectCompiledTransformer("c")
    tf.schema = """\
inputs:
  - name: item
    dtype:
      fields:
        - {name: x, dtype: int32}
        - {name: y, dtype: float64}
outputs:
  - name: result
    dtype:
      fields:
        - {name: x, dtype: int32}
        - {name: y, dtype: float64}
"""
    tf.code = """\
#include "public.h"
int transform(ItemStruct item, ResultStruct *result) {
    result->x = item.x + 1;
    result->y = item.y * 2.0;
    return 0;
}
"""
    dtype = np.dtype([("x", "int32"), ("y", "float64")], align=True)
    result = tf(item=np.array((4, 1.5), dtype=dtype)[()])
    assert result.dtype == dtype
    assert result["x"] == 5
    assert result["y"] == 3.0


def test_structured_array_with_fixed_field_shape():
    tf = DirectCompiledTransformer("c")
    tf.schema = """\
inputs:
  - name: points
    dtype:
      fields:
        - {name: xy, dtype: float64, shape: [2]}
        - {name: label, dtype: int32}
    shape: [N]
outputs:
  - name: shifted
    dtype:
      fields:
        - {name: xy, dtype: float64, shape: [2]}
        - {name: label, dtype: int32}
    shape: [N]
"""
    tf.code = """\
#include "public.h"
int transform(unsigned int N, const PointsStruct *points, ShiftedStruct *shifted) {
    for (unsigned int i = 0; i < N; i++) {
        shifted[i].xy[0] = points[i].xy[0] + 10.0;
        shifted[i].xy[1] = points[i].xy[1] - 1.0;
        shifted[i].label = points[i].label + 100;
    }
    return 0;
}
"""
    dtype = np.dtype([("xy", "float64", (2,)), ("label", "int32")], align=True)
    points = np.zeros(3, dtype=dtype)
    points["xy"] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    points["label"] = [7, 8, 9]

    result = tf(points=points)
    assert result.dtype == dtype
    np.testing.assert_allclose(
        result["xy"],
        np.array([[11.0, 1.0], [13.0, 3.0], [15.0, 5.0]]),
    )
    np.testing.assert_array_equal(result["label"], np.array([107, 108, 109]))


def test_multi_output_mixed_and_deepcell():
    schema = """\
inputs:
  - {name: a, dtype: int32}
  - {name: b, dtype: int32}
outputs:
  - {name: sum, dtype: int32}
  - {name: product, dtype: int32}
"""
    code = """\
#include <stdint.h>
int transform(int32_t a, int32_t b, int32_t *sum, int32_t *product) {
    *sum = a + b;
    *product = a * b;
    return 0;
}
"""
    tf = DirectCompiledTransformer("c")
    tf.schema = schema
    tf.code = code
    assert tf(a=2, b=3) == {"sum": 5, "product": 6}

    delayed = CompiledTransformer("c")
    delayed.schema = schema
    delayed.celltypes.result = "deepcell"
    delayed.code = code
    transformation = delayed(a=2, b=3)
    packed = transformation.run()
    assert set(packed) == {"sum", "product"}
    assert all(Checksum(value) for value in packed.values())

    direct = DirectCompiledTransformer("c")
    direct.schema = schema
    direct.celltypes.result = "deepcell"
    direct.code = code
    assert direct(a=2, b=3) == {"sum": 5, "product": 6}


@pytest.mark.skipif(not shutil.which("gfortran"), reason="gfortran required")
def test_multi_object_c_fortran():
    tf = DirectCompiledTransformer("c")
    tf.schema = ADD_SCHEMA
    tf.code = """\
#include <stdint.h>
extern void add_helper_(int32_t *a, int32_t *b, int32_t *result);
int transform(int32_t a, int32_t b, int32_t *result) {
    add_helper_(&a, &b, result);
    return 0;
}
"""
    obj = CompiledObject(language="fortran")
    obj.code = """\
subroutine add_helper(a,b,result)
use iso_c_binding
integer(c_int), intent(in) :: a,b
integer(c_int), intent(out) :: result
result = a + b
end subroutine
"""
    tf.objects.append(obj)
    assert tf(a=8, b=9) == 17


def test_non_native_endian_array_rejected():
    tf = DirectCompiledTransformer("c")
    tf.schema = """\
inputs:
  - {name: arr, dtype: int32, shape: [N]}
outputs:
  - {name: result, dtype: int32}
"""
    tf.code = """\
#include <stdint.h>
int transform(unsigned int N, const int32_t *arr, int32_t *result) {
    *result = arr[0];
    return 0;
}
"""
    dtype = ">i4" if np.dtype("int32").byteorder in ("<", "=") else "<i4"
    arr = np.array([1, 2], dtype=dtype)
    with pytest.raises(TypeError):
        tf(arr=arr)
