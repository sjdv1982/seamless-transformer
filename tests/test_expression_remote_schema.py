"""Real jobserver/DB round trip for the renamed Expression wire recipe."""
import os
import subprocess
import sys
import textwrap
import uuid

from test_run_transformation_cli import _write_remote_config


def test_expression_remote_schema_and_database_cache_hit(tmp_path):
    project = 'expression-schema-' + uuid.uuid4().hex
    _write_remote_config(tmp_path, backend='jobserver', project=project)
    script = textwrap.dedent('''
        import asyncio
        import sqlite3
        from pathlib import Path
        import seamless
        from seamless import Buffer, Expression
        import seamless_config
        from seamless_remote import buffer_remote, jobserver_remote
        from seamless.checksum.expression import get_expression_cache

        seamless_config.init()
        source = Buffer("schema hello", "str")
        source.tempref()
        cs = source.get_checksum()
        assert asyncio.run(buffer_remote.write_buffer(cs, source))
        first = Expression(cs, input_celltype="str", celltype="text")
        result = first.compute(execution="remote")
        assert result == Buffer("schema hello", "text").get_checksum()
        db_path, = Path("buffers").rglob("seamless.db")
        with sqlite3.connect(db_path) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(expression)")}
            assert "input_celltype" in columns and "celltype" in columns
            assert "target_celltype" not in columns
            row = db.execute("SELECT input_celltype, celltype, result FROM expression WHERE input_checksum = ?", (cs.hex(),)).fetchone()
            assert row == ("str", "text", result.hex()), row
        get_expression_cache().clear()
        async def forbid_dispatch(*args, **kwargs):
            raise AssertionError("second evaluation must use the database")
        jobserver_remote.run_expression = forbid_dispatch
        second = Expression(cs, input_celltype="str", celltype="text")
        assert second.compute(execution="remote") == result
        seamless.close()
        print("REMOTE_SCHEMA_AND_DB_CACHE_OK")
    ''')
    proc = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                          capture_output=True, text=True, timeout=90,
                          env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'REMOTE_SCHEMA_AND_DB_CACHE_OK' in proc.stdout
