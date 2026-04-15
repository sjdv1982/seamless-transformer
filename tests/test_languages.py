import pytest

from seamless_transformer.languages import (
    define_compiled_language,
    get_language,
    is_compiled_language,
)


def test_native_languages_registered():
    assert get_language("c").compilation.compiler == "gcc"
    assert get_language("cpp").compilation.compiler == "g++"
    assert get_language("fortran").compilation.compiler == "gfortran"
    assert get_language("rust").compilation.compiler == "rustc"
    assert not is_compiled_language("python")


def test_define_invalid_mode():
    with pytest.raises(ValueError):
        define_compiled_language(
            "bad-mode",
            {
                "compiler": "cc",
                "mode": "package",
                "options": [],
                "debug_options": [],
                "profile_options": [],
                "release_options": [],
                "compile_flag": "-c",
                "output_flag": "-o",
                "language_flag": "",
            },
        )


def test_define_custom_language():
    define_compiled_language(
        "custom-c",
        {
            "compiler": "cc",
            "mode": "object",
            "options": ["-O2"],
            "debug_options": ["-g"],
            "profile_options": [],
            "release_options": [],
            "compile_flag": "-c",
            "output_flag": "-o",
            "language_flag": "-x c",
        },
    )
    assert get_language("custom-c").compilation.options == ["-O2"]
