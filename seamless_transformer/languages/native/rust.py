from seamless_transformer.languages import define_compiled_language


define_compiled_language(
    name="rust",
    compilation={
        "compiler": "rustc",
        "mode": "archive",
        "options": ["--crate-type=staticlib"],
        "debug_options": ["--crate-type=staticlib"],
        "profile_options": [],
        "release_options": [],
        "compile_flag": "",
        "output_flag": "-o",
        "language_flag": "",
    },
)
