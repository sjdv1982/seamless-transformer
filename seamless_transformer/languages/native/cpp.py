from seamless_transformer.languages import define_compiled_language


define_compiled_language(
    name="cpp",
    compilation={
        "compiler": "g++",
        "mode": "object",
        "options": ["-O3", "-ffast-math", "-march=native", "-fPIC", "-fopenmp"],
        "debug_options": ["-g", "-fPIC"],
        "profile_options": [],
        "release_options": [],
        "compile_flag": "-c",
        "output_flag": "-o",
        "language_flag": "-x c++",
    },
)
