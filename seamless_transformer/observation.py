"""Transformation observation — an off-by-default record of what actually ran.

Design reference: ``seamless/attachments-and-mount-design.md`` §15 A0, where
**[MOD-14]** establishes the category: observability machinery added to the
product so that behaviour can be pinned *before* it is removed.  This is the
second instrument of that pair.  [MOD-14] records materialisation; this records
transformations — one line per transformation reaching an execution decision,
saying which transformation it was, whose it was, and whether the cache answered.

**Off by default, and cheap when off.**  :func:`is_observing` is a module-global
read; nothing is computed, formatted or opened unless a recording is running.
Call sites must guard any work they do purely to produce a record:

    if observation.is_observing():
        observation.observe(expensive_identity(), observation.CACHE_MISS, label=...)

**Why not count from the test side.**  The obvious alternative is to make the
transformer body itself append to a file, passing the path as a pin.  That works,
and it was what this suite did first, but the log path then becomes part of the
transformation identity — so the instrument perturbs the very property the tests
are about, and its failure mode is asymmetric in the wrong direction.  An
identity that accidentally *differs* produces a spurious re-execution: the count
goes up and the test fails loudly.  An identity that accidentally *collides*
produces a spurious cache hit: the count stays low, which reads as "nothing
re-executed" — a pass.  An instrument whose failure looks like success, sharing a
failure mode with the system under test, is the wrong instrument however elegant.
Observing from outside the transformation removes the whole question.

**The line format** is three tab-separated fields::

    <transformation checksum>\\t<label>\\t<cache-hit|cache-miss>

The *label* is who the transformation belonged to: a dotted node path for a
workflow-bound transformer (``tf``, ``sub.tf``), and for an unbound one an
identifier derived from its code — see :func:`default_label`.  It answers a
different question from the checksum: the checksum says *which computation*, the
label says *which transformer asked for it*, and the interesting rows are the
ones where those two disagree with expectation.

**Two recording sites, on purpose.**  ``transformation_cache`` records what goes
through the cache.  ``seamless_workflow.Context._derive_transformer`` records its
*direct* call to the Python callable, which never reaches the cache at all —
which is precisely the A0 defect, and is why a cache-side instrument alone would
report zero.  A1 deletes that call site, and its recording goes with it.

The environment variable ``SEAMLESS_TRANSFORMATION_OBSERVATION`` names the log
path, so a recording survives into worker processes when execution stops being
in-process.  :func:`start_observation` sets it for exactly that reason.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, NamedTuple


#: Outcome: the transformation was answered from a cache, local or remote.
CACHE_HIT = "cache-hit"

#: Outcome: no cache answered; the transformation reached execution.
CACHE_MISS = "cache-miss"

#: Names the log path.  Set by :func:`start_observation` so workers inherit it.
ENV_VAR = "SEAMLESS_TRANSFORMATION_OBSERVATION"


_lock = threading.Lock()
_path: str | None = None

#: Overrides the derived label for the duration of a block (see :func:`observed_as`).
_label: ContextVar[str | None] = ContextVar("seamless_observation_label", default=None)


# ------------------------------------------------------------------ switching


def observation_path() -> str | None:
    """The log path, or ``None`` when no recording is running."""

    if _path is not None:
        return _path
    return os.environ.get(ENV_VAR) or None


def is_observing() -> bool:
    """Whether anything would be recorded.  The guard for every call site."""

    return observation_path() is not None


def start_observation(path, *, propagate_to_workers: bool = True) -> None:
    """Begin recording to ``path``.  Idempotent replacement, not a stack."""

    global _path
    with _lock:
        _path = str(path)
        if propagate_to_workers:
            os.environ[ENV_VAR] = _path


def stop_observation() -> None:
    """End recording.  Leaves the file in place."""

    global _path
    with _lock:
        _path = None
        os.environ.pop(ENV_VAR, None)


@contextmanager
def observing(path) -> Iterator["TransformationObservations"]:
    """``with observing(tmp_path / "tf.log") as observations: ...``

    Restores whatever was there before, rather than clearing: a caller who had
    ``SEAMLESS_TRANSFORMATION_OBSERVATION`` set in the environment should not
    find it deleted because something ran a scoped recording inside.
    """

    global _path

    previous_path = _path
    previous_env = os.environ.get(ENV_VAR)
    start_observation(path)
    try:
        yield TransformationObservations(path)
    finally:
        with _lock:
            _path = previous_path
            if previous_env is None:
                os.environ.pop(ENV_VAR, None)
            else:
                os.environ[ENV_VAR] = previous_env


@contextmanager
def observed_as(label: str | None) -> Iterator[None]:
    """Label everything observed in this block.

    Used where the caller knows whose transformation it is and the recording
    site does not — the workflow layer, once its transformers submit through the
    cache rather than calling their callable directly.
    """

    token = _label.set(label)
    try:
        yield
    finally:
        _label.reset(token)


# ------------------------------------------------------------------- recording


def default_label(transformation_dict: dict[str, Any] | None) -> str:
    """Identify an unbound transformer from the transformation it submitted.

    There is no transformer id in the codebase to borrow, and ``id(obj)`` is not
    an identity in any useful sense.  What does identify a transformer across
    runs, and distinguishes it from a *different* transformer while remaining
    stable across its own varying inputs, is its **code checksum** — so that is
    what is recorded, shortened, as ``code:<first 12 hex>``.  A transformation
    dict always carries one; a label is therefore always available without any
    plumbing through the call chain.
    """

    if not transformation_dict:
        return "code:?"
    entry = transformation_dict.get("code")
    if not entry:
        return "code:?"
    try:
        checksum = entry[2]
    except (IndexError, TypeError):
        return "code:?"
    if checksum is None:
        return "code:?"
    text = checksum.hex() if hasattr(checksum, "hex") else str(checksum)
    return f"code:{text[:12]}"


def observe(
    tf_checksum,
    outcome: str,
    *,
    label: str | None = None,
    transformation_dict: dict[str, Any] | None = None,
) -> None:
    """Record one line.  A no-op when no recording is running.

    ``outcome`` is :data:`CACHE_HIT` or :data:`CACHE_MISS`.  ``label`` wins over
    the ambient :func:`observed_as` label, which wins over
    :func:`default_label`.

    Never raises: an instrument that can fail the run it observes is worse than
    a missing line, and a full disk during a test suite should not be reported as
    a contract violation.
    """

    path = observation_path()
    if path is None:
        return
    if label is None:
        label = _label.get()
    if label is None:
        label = default_label(transformation_dict)
    text = tf_checksum.hex() if hasattr(tf_checksum, "hex") else str(tf_checksum)
    line = f"{text}\t{label}\t{outcome}\n"
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        pass


# --------------------------------------------------------------------- reading


class Observation(NamedTuple):
    """One recorded line."""

    checksum: str
    label: str
    outcome: str


class TransformationObservations:
    """Reader for a recording.  Counts are the assertion surface."""

    def __init__(self, path) -> None:
        self.path = str(path)

    def entries(self) -> list[Observation]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                lines = [line.rstrip("\n") for line in handle if line.strip()]
        except FileNotFoundError:
            return []
        observations = []
        for line in lines:
            fields = line.split("\t")
            if len(fields) == 3:
                observations.append(Observation(*fields))
        return observations

    def _select(self, outcome: str, label: str | None) -> list[Observation]:
        return [
            entry
            for entry in self.entries()
            if entry.outcome == outcome and (label is None or entry.label == label)
        ]

    def hits(self, label: str | None = None) -> list[Observation]:
        return self._select(CACHE_HIT, label)

    def misses(self, label: str | None = None) -> list[Observation]:
        return self._select(CACHE_MISS, label)

    def count(self, label: str | None = None) -> int:
        """Every observation, hits and misses alike."""

        return len(
            [
                entry
                for entry in self.entries()
                if label is None or entry.label == label
            ]
        )

    def labels(self) -> set[str]:
        return {entry.label for entry in self.entries()}

    def checksums(self, label: str | None = None) -> list[str]:
        return [
            entry.checksum
            for entry in self.entries()
            if label is None or entry.label == label
        ]

    def __repr__(self) -> str:
        return f"TransformationObservations({self.path!r}, {self.entries()!r})"


__all__ = [
    "CACHE_HIT",
    "CACHE_MISS",
    "ENV_VAR",
    "Observation",
    "TransformationObservations",
    "default_label",
    "is_observing",
    "observation_path",
    "observe",
    "observed_as",
    "observing",
    "start_observation",
    "stop_observation",
]
