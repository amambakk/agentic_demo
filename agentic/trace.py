"""
The trace: a nested, timestamped log of everything the agent did.

Teaching point
--------------
An agent's control flow changes from run to run. That means a stack trace
or a print statement is not enough to debug it -- you need a record of
*which* step ran, with *which* arguments, how long it took, how many
attempts it needed, and what it observed. That record is the trace.

Everything printed by this project comes from the trace, not from stray
prints inside the logic. That separation is deliberate: the same run can
be rendered as a console tree or dumped as JSON for an eval harness.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator

# Stage names used across the project. Keeping them in one place means the
# renderer, the JSON dump and the tests all agree on the vocabulary.
STAGES = (
    "GOAL",
    "PLAN",
    "ACT",
    "OBSERVE",
    "REFLECT",
    "RESPOND",
    "MEMORY",
    "AGENT",
    "NOTE",
)


@dataclass
class Event:
    seq: int
    depth: int
    stage: str
    label: str
    detail: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    duration_ms: float | None = None
    status: str = "ok"  # ok | failed | recovered | skipped

    def as_dict(self) -> dict:
        return asdict(self)


class Tracer:
    """Collects Events and renders them as an indented tree."""

    def __init__(self, enabled: bool = True, stream=None) -> None:
        self.enabled = enabled
        self.events: list[Event] = []
        self._depth = 0
        self._seq = 0
        self._stream = stream
        self._t0 = time.perf_counter()

    # -- emitting ---------------------------------------------------------

    def event(self, stage: str, label: str, /, **detail: Any) -> Event:
        """Record a leaf event (no nested children, no duration)."""
        ev = Event(
            seq=self._next_seq(),
            depth=self._depth,
            stage=stage,
            label=label,
            detail=detail,
            started_at=time.perf_counter() - self._t0,
        )
        self.events.append(ev)
        self._render(ev)
        return ev

    @contextmanager
    def span(self, stage: str, label: str, /, **detail: Any) -> Iterator[Event]:
        """Record an event that has children and a duration.

        Usage:
            with tracer.span("ACT", "weather", city="Lagos") as ev:
                ...
                ev.detail["result"] = "..."
        """
        ev = Event(
            seq=self._next_seq(),
            depth=self._depth,
            stage=stage,
            label=label,
            detail=detail,
            started_at=time.perf_counter() - self._t0,
        )
        self.events.append(ev)
        self._render(ev, opening=True)
        self._depth += 1
        start = time.perf_counter()
        try:
            yield ev
        except Exception:
            ev.status = "failed"
            raise
        finally:
            ev.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self._depth -= 1

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # -- rendering --------------------------------------------------------

    def _render(self, ev: Event, opening: bool = False) -> None:
        if not self.enabled:
            return
        print(self.format_event(ev), file=self._stream)

    @staticmethod
    def format_event(ev: Event) -> str:
        indent = "   " * ev.depth
        marker = {"failed": " x", "recovered": " ~", "skipped": " -"}.get(ev.status, "")
        head = f"{indent}{ev.stage:<8}{marker} {ev.label}"
        if ev.detail:
            head += "  " + _fmt_detail(ev.detail)
        return head

    # -- export -----------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        return json.dumps([e.as_dict() for e in self.events], indent=indent, default=str)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    def summary(self) -> dict:
        """Cheap trajectory metrics -- the numbers an eval harness wants."""
        tool_calls = [e for e in self.events if e.stage == "ACT"]
        return {
            "events": len(self.events),
            "tool_calls": len(tool_calls),
            "failed_calls": sum(1 for e in tool_calls if e.status == "failed"),
            "recovered_calls": sum(1 for e in tool_calls if e.status == "recovered"),
            "total_ms": round(
                sum(e.duration_ms or 0 for e in self.events if e.depth == 0), 2
            ),
        }


def _fmt_detail(detail: dict) -> str:
    parts = []
    for key, value in detail.items():
        text = str(value)
        if len(text) > 90:
            text = text[:87] + "..."
        parts.append(f"{key}={text}")
    return "(" + ", ".join(parts) + ")"


def rule(title: str = "", width: int = 78, char: str = "=") -> None:
    """Console separator used by the lessons."""
    if not title:
        print(char * width)
        return
    pad = max(0, width - len(title) - 3)
    print(f"{char * 2} {title} {char * pad}")
