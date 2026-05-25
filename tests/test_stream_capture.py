from __future__ import annotations

import io

from seamless_transformer.stream_capture import _StreamingTap


def test_streaming_tap_mirrors_sink_and_truncates_from_head() -> None:
    sink = io.StringIO()
    chunks = []
    tap = _StreamingTap(
        stream_name="stdout",
        sink=sink,
        notifier=chunks.append,
        max_payload=5,
        min_interval=100.0,
    )
    try:
        assert tap.write("hello world") == len("hello world")
        tap.flush()
    finally:
        tap.close()

    assert sink.getvalue() == "hello world"
    assert chunks[0]["kind"] == "stream"
    assert chunks[0]["stream"] == "stdout"
    assert chunks[0]["text"] == "world"
    assert chunks[0]["truncated_head_bytes"] == len("hello ")


def test_streaming_tap_coalesces_until_forced_flush() -> None:
    sink = io.StringIO()
    chunks = []
    tap = _StreamingTap(
        stream_name="stderr",
        sink=sink,
        notifier=chunks.append,
        max_payload=100,
        min_interval=100.0,
    )
    try:
        tap.write("alpha")
        assert len(chunks) == 1
        tap.write(" beta")
        assert len(chunks) == 1
        tap.flush()
    finally:
        tap.close()

    assert sink.getvalue() == "alpha beta"
    assert len(chunks) == 2
    assert chunks[0]["stream"] == "stderr"
    assert chunks[0]["text"] == "alpha"
    assert chunks[1]["text"] == " beta"
