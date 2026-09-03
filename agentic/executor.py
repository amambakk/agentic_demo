"""
The Act + Observe half of the loop: call the tool, survive it, grade it.

Teaching point
--------------
Recovery policy belongs in one place, declared up front:

    retry      only for `transient` failures, bounded, with backoff
    fallback   a declared map primary_tool -> [alternates], not improvised
    give up    record the failure as a gap and let the critic decide

Improvised recovery ("if it fails, try something else") produces agents
that behave differently every run and cannot be tested.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .errors import ToolError
from .planner import Step
from .tools import ToolRegistry, ToolResult
from .trace import Tracer

# Declared, inspectable recovery routes.
DEFAULT_FALLBACKS: dict[str, list[str]] = {
    "rag_search": ["web_search"],   # corrective RAG: internal -> external
    "weather": ["web_search"],      # no coverage -> try the news snapshot
    "web_search": ["rag_search"],   # external down -> curated knowledge
}


@dataclass
class Observation:
    """What the agent learned from one step. The unit of evidence."""

    step: Step
    tool: str
    ok: bool
    summary: str
    data: Any = None
    citations: list[str] = field(default_factory=list)
    grade: str = "strong"
    attempts: int = 1
    error: dict | None = None
    recovered_with: str | None = None
    alternates_tried: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.ok and self.grade != "weak"

    def line(self) -> str:
        if not self.ok:
            return f"[FAILED] {self.step.intent}: {self.error and self.error['message']}"
        prefix = f"[via {self.recovered_with}] " if self.recovered_with else ""
        return f"{prefix}{self.summary}"


class Executor:
    def __init__(
        self,
        registry: ToolRegistry,
        tracer: Tracer,
        max_attempts: int = 3,
        backoff_s: float = 0.02,
        fallbacks: dict[str, list[str]] | None = None,
    ) -> None:
        self.registry = registry
        self.tracer = tracer
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.fallbacks = DEFAULT_FALLBACKS if fallbacks is None else fallbacks
        self.dead_tools: set[str] = set()

    # -- public API -------------------------------------------------------

    def run_step(self, step: Step) -> Observation:
        with self.tracer.span("ACT", step.tool, **{k: str(v)[:60] for k, v in step.args.items()}) as ev:
            obs = self._attempt_with_recovery(step, ev)
            ev.detail["result"] = obs.summary if obs.ok else "FAILED"
            ev.status = (
                "failed" if not obs.ok
                else "recovered" if (obs.recovered_with or obs.attempts > 1)
                else "ok"
            )
        self.tracer.event(
            "OBSERVE",
            step.tool,
            grade=obs.grade,
            usable=obs.usable,
            detail=obs.line(),
        )
        return obs

    # -- internals --------------------------------------------------------

    def _attempt_with_recovery(self, step: Step, ev) -> Observation:
        primary_error: ToolError | None = None

        if step.tool not in self.dead_tools:
            try:
                result, attempts = self._call_with_retry(step.tool, step.args)
                obs = self._observation(step, step.tool, result, attempts)
                # Corrective RAG: a *successful* call that returned weakly
                # relevant evidence is a soft failure, and soft failures are
                # where confident wrong answers come from. Try the alternate
                # source before accepting it.
                if obs.grade == "weak":
                    return self._try_better_source(step, obs)
                return obs
            except ToolError as exc:
                primary_error = exc
                if exc.kind == "unavailable":
                    # Remember the outage so later steps skip this dependency.
                    self.dead_tools.add(step.tool)
                    self.tracer.event("NOTE", "marking tool dead", tool=step.tool)
        else:
            primary_error = ToolError(
                "tool already marked dead in this run",
                kind="unavailable",
                tool=step.tool,
            )
            self.tracer.event("NOTE", "skipping dead tool", tool=step.tool)

        # Primary failed. Walk the declared fallback chain.
        for alt in self.fallbacks.get(step.tool, []):
            if alt in self.dead_tools or not self.registry.has(alt):
                continue
            self.tracer.event(
                "NOTE", "falling back",
                frm=step.tool, to=alt, because=primary_error.kind,
            )
            alt_args = self._translate_args(step, alt)
            try:
                result, attempts = self._call_with_retry(alt, alt_args)
                obs = self._observation(step, alt, result, attempts)
                obs.recovered_with = alt
                return obs
            except ToolError as exc:
                self.tracer.event("NOTE", "fallback failed", tool=alt, kind=exc.kind)

        return Observation(
            step=step, tool=step.tool, ok=False,
            summary=f"{step.tool} failed and no fallback succeeded",
            error=primary_error.as_dict() if primary_error else None,
            grade="none",
        )

    def _try_better_source(self, step: Step, weak: Observation) -> Observation:
        """Given weakly relevant evidence, see if another source does better.

        Note the policy: we only *upgrade*. If the alternate source is also
        weak, we keep the original and leave the weak grade in place so the
        critic still sees it. Silently swapping weak evidence for different
        weak evidence would hide the problem rather than solve it.
        """
        for alt in self.fallbacks.get(step.tool, []):
            if alt in self.dead_tools or not self.registry.has(alt):
                continue
            self.tracer.event(
                "NOTE", "weak evidence; trying alternate source",
                frm=step.tool, to=alt,
            )
            weak.alternates_tried.append(alt)
            try:
                result, attempts = self._call_with_retry(alt, self._translate_args(step, alt))
            except ToolError as exc:
                self.tracer.event("NOTE", "alternate source failed", tool=alt, kind=exc.kind)
                continue
            if result.grade == "strong":
                obs = self._observation(step, alt, result, attempts)
                obs.recovered_with = alt
                return obs
            self.tracer.event("NOTE", "alternate source also weak; keeping original", tool=alt)
        return weak

    def _call_with_retry(self, tool_name: str, args: dict) -> tuple[ToolResult, int]:
        tool = self.registry.get(tool_name)
        last: ToolError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return tool.run(**args), attempt
            except ToolError as exc:
                last = exc
                if not exc.retryable:
                    # Retrying invalid_input or not_found just wastes budget:
                    # identical arguments produce an identical failure.
                    self.tracer.event(
                        "NOTE", "not retryable",
                        tool=tool_name, kind=exc.kind, msg=exc.message,
                    )
                    raise
                if attempt < self.max_attempts:
                    delay = self.backoff_s * (2 ** (attempt - 1))
                    self.tracer.event(
                        "NOTE", "retrying",
                        tool=tool_name, attempt=attempt + 1,
                        after_ms=round(delay * 1000, 1), because=exc.message,
                    )
                    time.sleep(delay)
        raise last  # type: ignore[misc]

    @staticmethod
    def _translate_args(step: Step, alt_tool: str) -> dict:
        """Arguments do not transfer between tools unchanged.

        A `weather(city='Kano')` step becomes `web_search(query='Kano weather')`.
        Forgetting this translation is a classic fallback bug: the alternate
        tool gets called with a keyword it does not understand.
        """
        if alt_tool in ("web_search", "rag_search"):
            if "query" in step.args:
                return {"query": step.args["query"]}
            if "city" in step.args:
                return {"query": f"{step.args['city']} weather"}
            return {"query": step.intent}
        return dict(step.args)

    @staticmethod
    def _observation(step: Step, tool: str, result: ToolResult, attempts: int) -> Observation:
        return Observation(
            step=step, tool=tool, ok=True,
            summary=result.summary, data=result.data,
            citations=result.citations, grade=result.grade, attempts=attempts,
        )
