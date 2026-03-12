from seamless import Buffer

from seamless_transformer.execute_bash import execute_bash


def _register_bytes(data: bytes) -> str:
    buffer = Buffer(data)
    checksum = buffer.get_checksum()
    buffer.tempref()
    return checksum.hex()


def test_execute_bash_directory_input_deepfolder_contents():
    deepfolder = {
        "a.txt": _register_bytes(b"apples\npears\noranges\nbananas\npineapples\n"),
        "b.txt": _register_bytes(b"5\n12\n7\n4\n15\n"),
    }
    bashcode = (
        "paste data/b.txt data/a.txt "
        "| awk 'NF == 2 && $1 > 10{print $2}' > RESULT"
    )
    result = execute_bash(
        bashcode,
        ["data"],
        "",
        {"data": deepfolder},
        {
            "data": {
                "mode": "directory",
                "filesystem": False,
                "hash_pattern": {"*": "##"},
            }
        },
        "bytes",
    )
    assert result.decode().strip().splitlines() == ["pears", "pineapples"]


def test_execute_bash_directory_input_plain_mapping():
    plain_mapping = {"file.txt": "hello world\n"}
    bashcode = "cat data/file.txt > RESULT"
    result = execute_bash(
        bashcode,
        ["data"],
        "",
        {"data": plain_mapping},
        {"data": {"mode": "directory", "filesystem": False}},
        "bytes",
    )
    assert result.decode() == "hello world\n"
