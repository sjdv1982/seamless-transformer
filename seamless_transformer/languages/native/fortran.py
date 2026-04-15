from seamless_transformer.languages import define_compiled_language


define_compiled_language(
    name="fortran",
    compilation={
        "compiler": "gfortran",
        "mode": "object",
        "options": [
            "-O3",
            "-fno-automatic",
            "-fcray-pointer",
            "-ffast-math",
            "-march=native",
            "-fPIC",
        ],
        "debug_options": ["-g", "-fPIC"],
        "profile_options": [],
        "release_options": [],
        "compile_flag": "-c",
        "output_flag": "-o",
        "language_flag": "-x f95",
    },
)
