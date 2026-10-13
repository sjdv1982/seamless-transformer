from seamless import Buffer, Checksum

from seamless_transformer.execute_bash import execute_bash


def _store_bytes(data: bytes, buffer_dir) -> Checksum:
    buffer = Buffer(data)
    checksum = Checksum(buffer.get_checksum())
    (buffer_dir / checksum.hex()).write_bytes(data)
    return checksum


def test_execute_bash_directory_input_deepfolder_contents(tmp_path, monkeypatch):
    buffer_dir = tmp_path / "read-buffer"
    buffer_dir.mkdir()
    from seamless_transformer import transformation_namespace

    monkeypatch.setattr(
        transformation_namespace, "_get_read_folder_directories", lambda: [str(buffer_dir)]
    )
    deepfolder = {
        "a.txt": _store_bytes(
            b"apples\npears\noranges\nbananas\npineapples\n", buffer_dir
        ),
        "b.txt": _store_bytes(b"5\n12\n7\n4\n15\n", buffer_dir),
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
                "celltype": "deepfolder",
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
