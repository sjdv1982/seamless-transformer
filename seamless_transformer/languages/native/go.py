from seamless_transformer.languages import define_compiled_language


define_compiled_language(
    name="go",
    compilation={
        "compiler": "go",
        "compile_flag": "build",
        "mode": "package",
        "extension": "go",
        "options": ["-buildmode=c-archive"],
        "debug_options": ["-buildmode=c-archive", "-gcflags=all=-N -l"],
        "profile_options": [],
        "release_options": [],
        "output_flag": "-o",
        "language_flag": "",
        "include_flag": "",
    },
)
