"""TR1 (4 Nov, report-only): the per-question retrieval and selection trace.

Why each section became a candidate, how it was scored, which step picked or added it, and which step left it out - the
facts that `section_promotion` notes and `Selection.to_dict` record only in part (no lexical, reranker or companion
scores; no routing decision; no expanded search terms; no record of a companion that was considered and refused).

The trace only reads values the pipeline has already computed; it never scores, queries or decides anything itself, so
an answer is byte-identical with TR1 on or off (tests/test_trace_tr1.py replays this). With TR1 off nothing is recorded
and the answer metadata has no "trace" key. Every recording call is guarded: a failure inside the trace is dropped and
never reaches the answer.

    tr = trace.begin()                      # pipeline._answer_once: a new trace for this question (None with TR1 off)
    trace.event("candidates", "branch", branch="programme", programme="Dietary supplements")
    trace.event("companions", "refused", step="MS1", section="111.75", score=0.62, reason="below MS1_MIN_SCORE 0.7")
    metadata["trace"] = tr.to_dict()

Steps used: query, candidates, rank, pick, companions, selection.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

from regulus.rag.validator_switches import enabled

log = logging.getLogger(__name__)

VERSION = 1
MAX_EVENTS = 400          # a runaway loop cannot grow a record without bound
_ACTIVE: ContextVar["Trace | None"] = ContextVar("regulus_tr1_trace", default=None)


def _plain(value: Any, depth: int = 0) -> Any:
    """JSON-safe copy: tuples and sets as lists, floats to 4 places, unknown objects as their str()."""
    if depth > 6:
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [_plain(v, depth + 1) for v in items]
    return str(value)


class Trace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.dropped = 0

    def event(self, step: str, action: str, **data: Any) -> None:
        if len(self.events) >= MAX_EVENTS:
            self.dropped += 1
            return
        self.events.append({"step": step, "action": action, **{k: _plain(v) for k, v in data.items()}})

    def to_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "events": list(self.events), **({"events_dropped": self.dropped} if self.dropped else {})}


def begin() -> Trace | None:
    """A new trace for one question when TR1 is on (it replaces any earlier one in this context); None otherwise."""
    tr = Trace() if enabled("TR1") else None
    _ACTIVE.set(tr)
    return tr


def active() -> Trace | None:
    return _ACTIVE.get()


def event(step: str, action: str, **data: Any) -> None:
    """Record one event on the active trace; a no-op without one, and never raises."""
    tr = _ACTIVE.get()
    if tr is None:
        return
    try:
        tr.event(step, action, **data)
    except Exception:  # noqa: BLE001 - report-only: a trace failure never reaches the answer
        log.debug("TR1 event dropped", exc_info=True)


def lazy(step: str, action: str, build) -> None:
    """event() whose data is built by `build()` (a dict) only when a trace is active, inside the guard - the form used
    at every pipeline hook, so TR1 off costs nothing and a hook's own expression can never raise into the pipeline."""
    tr = _ACTIVE.get()
    if tr is None:
        return
    try:
        data = build()
        if data is not None:   # None: nothing to record this time
            tr.event(step, action, **data)
    except Exception:  # noqa: BLE001 - report-only: a trace failure never reaches the answer
        log.debug("TR1 event dropped", exc_info=True)
