"""Real jobserver/DB round trip for the renamed Expression wire recipe."""
import os
import subprocess
import sys
import textwrap
import uuid
import pytest

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


def test_auto_expression_dispatches_hashserver_only_input_to_jobserver(tmp_path):
    project = 'expression-auto-location-' + uuid.uuid4().hex
    _write_remote_config(tmp_path, backend='jobserver', project=project)
    script = textwrap.dedent('''
        import asyncio
        import seamless
        from seamless import Buffer, Checksum, Expression
        import seamless_config
        from seamless.caching.buffer_cache import get_buffer_cache
        from seamless.checksum import expression as expression_mod
        from seamless.checksum.cached_calculate_checksum import checksum_cache
        from seamless_remote import buffer_remote, jobserver_remote

        seamless_config.init()
        try:
            source = Buffer({"value": "from jobserver"}, "plain")
            source_checksum = source.get_checksum()
            assert asyncio.run(buffer_remote.write_buffer(source_checksum, source))

            cache = get_buffer_cache()
            with cache.lock:
                cache.weak_cache.pop(source_checksum, None)
                cache.strong_cache.pop(source_checksum, None)
            checksum_cache.pop(source_checksum, None)
            expression_mod._expression_result_buffers.pop(source_checksum, None)
            expression_mod.get_expression_cache().clear()

            dispatches = []
            original_run_expression = jobserver_remote.run_expression
            async def record_dispatch(*args, **kwargs):
                dispatches.append((args, kwargs))
                return await original_run_expression(*args, **kwargs)
            jobserver_remote.run_expression = record_dispatch

            expression = Expression(
                source_checksum,
                "value",
                input_celltype="plain",
                celltype="str",
            )
            result = expression.compute(execution="auto")
            assert result == Buffer("from jobserver", "str").get_checksum()
            assert len(dispatches) == 1, dispatches
            args, kwargs = dispatches[0]
            assert Checksum(args[0]) == source_checksum
            assert args[1:] == ("value", "plain", "str")
            assert kwargs == {"scratch": True}
        finally:
            seamless.close()
        print("AUTO_EXPRESSION_JOBSERVER_OK")
    ''')
    proc = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                          capture_output=True, text=True, timeout=90,
                          env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'AUTO_EXPRESSION_JOBSERVER_OK' in proc.stdout


@pytest.mark.parametrize("backend", ["jobserver", "daskserver"])
def test_jobserver_merges_duplicate_requests(tmp_path, backend):
    project = 'expression-duplicate-requests-' + uuid.uuid4().hex
    _write_remote_config(tmp_path, backend=backend, project=project)
    script = textwrap.dedent('''
        import asyncio
        import json
        import os
        import subprocess
        import sys
        from urllib.request import urlopen

        import seamless
        from seamless import Buffer
        import seamless_config
        from seamless_remote import buffer_remote, jobserver_remote

        child_script = r"""
        import seamless
        from seamless import Expression
        import seamless_config
        seamless_config.init()
        try:
            result = Expression(
                __import__('seamless').Checksum(__import__('os').environ['SOURCE_CHECKSUM']),
                'value',
                input_celltype='plain',
                celltype='str',
            ).compute(execution='remote')
            print(result.hex())
        finally:
            seamless.close()
        """

        seamless_config.init()
        try:
            source = Buffer({"value": "x" * 10_000_000}, "plain")
            source_checksum = source.get_checksum()
            assert asyncio.run(buffer_remote.write_buffer(source_checksum, source))
            def evaluation_count():
                from seamless_dask.transformer_client import get_seamless_dask_client
                client = get_seamless_dask_client()
                if client is not None:
                    def count():
                        from seamless.checksum import expression
                        return expression._expression_evaluations
                    return sum(client.client.run(count).values())
                with urlopen(jobserver_remote._jobserver_clients[0].url + "/status", timeout=10) as response:
                    return json.load(response)["expression_evaluations"]
            before = evaluation_count()

            env = {**os.environ, "SOURCE_CHECKSUM": source_checksum.hex()}
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", child_script],
                    cwd=os.getcwd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                for _ in range(2)
            ]
            outputs = [process.communicate(timeout=90) for process in processes]
            for process, (stdout, stderr) in zip(processes, outputs):
                assert process.returncode == 0, stdout + stderr
            checksums = [stdout.strip().splitlines()[-1] for stdout, _ in outputs]
            assert checksums[0] == checksums[1]

            after = evaluation_count()
            assert after - before == 1, (before, after)
        finally:
            seamless.close()
        print("DUPLICATE_EXPRESSION_REQUESTS_OK")
    ''')
    proc = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                          capture_output=True, text=True, timeout=180,
                          env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'DUPLICATE_EXPRESSION_REQUESTS_OK' in proc.stdout


def test_run_resolves_jobserver_result_through_hashserver(tmp_path):
    project = 'expression-run-hashserver-' + uuid.uuid4().hex
    _write_remote_config(tmp_path, backend='jobserver', project=project)
    script = textwrap.dedent('''
        import asyncio
        import seamless
        from seamless import Buffer, CacheMissError, Checksum, Expression
        import seamless_config
        from seamless.caching.buffer_cache import get_buffer_cache
        from seamless.checksum import expression as expression_mod
        from seamless.checksum.cached_calculate_checksum import checksum_cache
        from seamless_remote import buffer_remote

        def drop_buffer(checksum):
            checksum = Checksum(checksum)
            cache = get_buffer_cache()
            with cache.lock:
                cache.weak_cache.pop(checksum, None)
                cache.strong_cache.pop(checksum, None)
            checksum_cache.pop(checksum, None)
            expression_mod._expression_result_buffers.pop(checksum, None)

        seamless_config.init()
        try:
            source = Buffer({"value": "from hashserver"}, "plain")
            source_checksum = source.get_checksum()
            assert asyncio.run(buffer_remote.write_buffer(source_checksum, source))
            expression = Expression(
                source_checksum,
                "value",
                input_celltype="plain",
                celltype="str",
            )
            result_checksum = expression.compute(execution="remote")

            drop_buffer(result_checksum)
            assert expression.run() == "from hashserver"

            # run() materializes an unreachable result from its input; only with
            # the input reachable nowhere does it surface CacheMissError, on the result.
            drop_buffer(result_checksum)
            drop_buffer(source_checksum)
            del source
            async def no_buffer(checksum):
                return None
            async def no_lengths(checksums):
                return [None for _ in checksums]
            buffer_remote.get_buffer = no_buffer
            buffer_remote.get_buffer_lengths = no_lengths
            try:
                expression.run()
            except CacheMissError as exc:
                assert exc.checksum == result_checksum
            else:
                raise AssertionError("an input reachable nowhere must surface CacheMissError")
        finally:
            seamless.close()
        print("EXPRESSION_RUN_HASHSERVER_OK")
    ''')
    proc = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                          capture_output=True, text=True, timeout=90,
                          env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'EXPRESSION_RUN_HASHSERVER_OK' in proc.stdout


@pytest.mark.parametrize("backend", ["jobserver", "daskserver"])
def test_jobserver_evaluation_stores_result_hash_type(tmp_path, backend):
    project = 'expression-result-hashtype-' + uuid.uuid4().hex
    _write_remote_config(tmp_path, backend=backend, project=project)
    script = textwrap.dedent('''
        import asyncio
        import sqlite3
        import time
        from pathlib import Path

        import seamless
        from seamless import Buffer, Expression
        import seamless_config
        from seamless.checksum.hash_type import HashType
        from seamless_remote import buffer_remote

        seamless_config.init()
        try:
            source = Buffer({"value": "classified result"}, "plain")
            source_checksum = source.get_checksum()
            assert asyncio.run(buffer_remote.write_buffer(source_checksum, source))
            expression = Expression(
                source_checksum,
                "value",
                input_celltype="plain",
                celltype="str",
            )
            result_checksum = expression.compute(execution="remote")
            expected = HashType.from_buffer(b'"classified result"').word
            db_path, = Path("buffers").rglob("seamless.db")
            deadline = time.monotonic() + 5
            row = None
            while time.monotonic() < deadline:
                with sqlite3.connect(db_path) as db:
                    row = db.execute(
                        "SELECT hash_type FROM hash_type WHERE checksum = ?",
                        (result_checksum.hex(),),
                    ).fetchone()
                if row is not None:
                    break
                time.sleep(0.05)
            assert row == (expected,), row
            from seamless.checksum.hash_type import get_hash_type_cache
            from seamless.checksum.hash_type_validation import ensure_hash_type
            from seamless.caching.buffer_cache import get_buffer_cache
            from seamless_remote import jobserver_remote
            from urllib.request import urlopen
            import json
            from seamless_dask.transformer_client import get_seamless_dask_client
            client = get_seamless_dask_client()
            if client is not None:
                def count():
                    from seamless.checksum import expression
                    return expression._expression_evaluations
                assert sum(client.client.run(count).values()) == 1
                from seamless.checksum import expression as expression_mod
                assert expression_mod._expression_evaluations == 0

            get_hash_type_cache().clear()
            cache = get_buffer_cache()
            with cache.lock:
                cache.weak_cache.pop(result_checksum, None)
                cache.strong_cache.pop(result_checksum, None)
            assert ensure_hash_type(result_checksum).word == expected

        finally:
            seamless.close()
        print("JOBSERVER_RESULT_HASHTYPE_OK")
    ''')
    proc = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                          capture_output=True, text=True, timeout=90,
                          env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'JOBSERVER_RESULT_HASHTYPE_OK' in proc.stdout


def test_daskserver_jobserver_boundary_keeps_buffers_on_worker(tmp_path):
    project = 'expression-dask-boundary-' + uuid.uuid4().hex
    _write_remote_config(tmp_path, backend='daskserver', project=project)
    script = textwrap.dedent('''
        import asyncio
        import json
        import seamless
        import seamless_config
        import jobserver
        from seamless import Buffer, Checksum, CacheMissError
        from seamless_remote import buffer_remote
        from seamless.checksum.expression import _get_local_buffer
        from seamless.error_envelope import decode_error

        class Request:
            def __init__(self, checksum):
                self.checksum = checksum
            async def json(self):
                return dict(input_checksum=self.checksum.hex(), path="value",
                            input_celltype="plain", celltype="str")

        seamless_config.init()
        try:
            source = Buffer({"value": "worker boundary"}, "plain")
            checksum = source.get_checksum()
            assert asyncio.run(buffer_remote.write_buffer(checksum, source))
            server = jobserver.JobServer("127.0.0.1", 0)
            original = Checksum.resolution
            async def forbidden(*args, **kwargs):
                raise AssertionError("HTTP process downloaded an Expression buffer")
            Checksum.resolution = forbidden
            try:
                response = asyncio.run(server._run_expression(Request(checksum)))
                missing = Checksum("b" * 64)
                failure = asyncio.run(server._run_expression(Request(missing)))
            finally:
                Checksum.resolution = original
            assert response.status == 200
            payload = json.loads(response.text)
            assert set(payload) == {"result_checksum"}, payload
            result = Checksum(payload["result_checksum"])
            try:
                _get_local_buffer(result)
            except CacheMissError:
                pass
            else:
                raise AssertionError("Result buffer reached the HTTP process")
            assert result.resolve("str") == "worker boundary"
            error = decode_error(json.loads(failure.text))
            assert failure.status == 200
            assert isinstance(error, CacheMissError)
            assert isinstance(error.args[0], Checksum)
            assert error.args[0] == missing
        finally:
            seamless.close()
        print("DASK_HTTP_BOUNDARY_OK")
    ''')
    proc = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                          capture_output=True, text=True, timeout=90,
                          env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'DASK_HTTP_BOUNDARY_OK' in proc.stdout
