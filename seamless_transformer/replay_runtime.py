"""Neutral replay runtime hooks.

This module intentionally has no dependency on seamless-share. Runtime packages
can import it unconditionally; when SEAMLESS_REPLAY_MODE is absent it is inert.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextvars
import json
import os
from pathlib import Path
from typing import Any


_driver_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "seamless_replay_driver_stack", default=()
)


@dataclass(frozen=True)
class ReplayRuntimeConfig:
    artifact: str
    bufferdir: str
    auth: str
    driver_cache: str
    allow_remote: bool
    event_path: str
    config_mode: str
    config_path: str | None = None


def active() -> bool:
    return os.environ.get("SEAMLESS_REPLAY_MODE") == "1"


def config() -> ReplayRuntimeConfig:
    return ReplayRuntimeConfig(
        artifact=os.environ.get("SEAMLESS_REPLAY_ARTIFACT", ""),
        bufferdir=os.environ.get("SEAMLESS_REPLAY_BUFFERDIR", ""),
        auth=os.environ.get("SEAMLESS_REPLAY_AUTH", ""),
        driver_cache=os.environ.get("SEAMLESS_REPLAY_DRIVER_CACHE", "bypass"),
        allow_remote=os.environ.get("SEAMLESS_REPLAY_ALLOW_REMOTE") == "1",
        event_path=os.environ.get("SEAMLESS_REPLAY_REPORT_EVENTS", ""),
        config_mode=os.environ.get("SEAMLESS_REPLAY_CONFIG_MODE", "synthesized"),
        config_path=os.environ.get("SEAMLESS_REPLAY_CONFIG"),
    )


def driver_context() -> list[str]:
    return list(_driver_stack.get())


def push_driver(tf_checksum: str):
    stack = _driver_stack.get()
    token = _driver_stack.set((*stack, tf_checksum))
    return token


def pop_driver(token) -> None:
    _driver_stack.reset(token)


def emit(event: dict[str, Any]) -> None:
    if not active():
        return
    event_path = config().event_path
    if not event_path:
        return
    payload = dict(event)
    payload.setdefault("driver_context", driver_context())
    path = Path(event_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def cache_lookup(tf_checksum, *, is_driver=False, transformation_dict=None, tf_dunder=None, script_position=None):
    emit(
        {
            "event": "cache_lookup",
            "tf_checksum": tf_checksum,
            "is_driver": is_driver,
            "script_position": script_position,
        }
    )


def materialization_request(checksum, *, requested_by=None, script_position=None, intent=None):
    emit(
        {
            "event": "materialization_request",
            "checksum": checksum,
            "requested_by": requested_by,
            "script_position": script_position,
            "intent": intent,
        }
    )


def fingertip_request(consumer_tf_checksum, missing_input_checksum, producer_tf_checksum, script_position=None):
    emit(
        {
            "event": "fingertip_request",
            "consumer_tf_checksum": consumer_tf_checksum,
            "missing_input_checksum": missing_input_checksum,
            "producer_tf_checksum": producer_tf_checksum,
            "script_position": script_position,
        }
    )


def remote_dispatch(backend, dispatched_work, script_position=None):
    emit(
        {
            "event": "remote_delegation_observed",
            "backend": backend,
            "dispatched_work": dispatched_work,
            "script_position": script_position,
        }
    )


def transformation_started(tf_checksum, *, is_driver=False):
    emit({"event": "transformation_started", "tf_checksum": tf_checksum, "is_driver": is_driver})


def transformation_finished(tf_checksum, *, is_driver=False, observed_cost_ms=0, cache_hit=False):
    emit(
        {
            "event": "transformation_finished",
            "tf_checksum": tf_checksum,
            "is_driver": is_driver,
            "observed_cost_ms": observed_cost_ms,
            "cache_hit": cache_hit,
        }
    )
