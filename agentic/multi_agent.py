"""
Multi-agent collaboration over a shared blackboard.

Roles
-----
    Researcher  read-only retrieval tools. Gathers evidence with citations.
    Analyst     calculator + weather. Produces numbers, never prose claims.
    Critic      no tools at all. Reviews the blackboard and lists gaps.
    Writer      no tools at all. Composes the answer from posted evidence.

Teaching points
---------------
1. Specialisation is enforced by the *toolbox*, not by instructions. The
   Researcher literally cannot do arithmetic because `calculator` is not in
   its registry. Constraining capability beats asking nicely.

2. The blackboard is shared, inspectable state. Compare this with
   point-to-point message passing, where you have to reconstruct who knew
   what from a pile of logs.

3. Handoffs lose fidelity. Every entry posted here keeps its citations and
   its origin agent, so the Writer composes from structured records rather
   than from a paraphrase of a paraphrase.

4. This is not automatically better than one agent with all five tools. It
   costs more calls and adds coordination failure modes. The gain is the
   independent review step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .agent import Agent, AgentResult, Composer
from .memory import EpisodicMemory
from .planner import DecompositionPlanner, KeywordRouter
from .tools import ToolRegistry
from .trace import Tracer


@dataclass
class Entry:
    """One posted finding. Structured, attributed, cited."""

    author: str
    kind: str            # evidence | number | critique | answer
    content: str
    citations: list[str] = field(default_factory=list)
    ok: bool = True


class Blackboard:
    """Shared workspace. Every agent reads and writes here."""

    def __init__(self, goal: str, tracer: Tracer) -> None:
        self.goal = goal
        self.entries: list[Entry] = []
        self.tracer = tracer

    def post(self, entry: Entry) -> Entry:
        self.entries.append(entry)
        self.tracer.event(
            "NOTE", f"{entry.author} posted {entry.kind}",
            content=entry.content[:80], cites=len(entry.citations),
        )
        return entry

    def by_kind(self, kind: str) -> list[Entry]:
        return [e for e in self.entries if e.kind == kind]

    def digest(self) -> str:
        return "\n".join(f"[{e.author}/{e.kind}] {e.content}" for e in self.entries)


class Specialist:
    """A role: a name, a narrowed toolbox, and one job."""

    def __init__(self, name: str, registry: ToolRegistry, tracer: Tracer, tools: list[str]) -> None:
        self.name = name
        self.tracer = tracer
        self.registry = registry.subset(tools)
        self.agent = Agent(
            registry=self.registry,
            tracer=tracer,
            planner=DecompositionPlanner(KeywordRouter()),
            episodic=EpisodicMemory(),      # sub-agents get their own memory
            name=name,
            max_iterations=1,               # sub-agents do not self-loop
        )

    def work(self, task: str, board: Blackboard) -> AgentResult:
        with self.tracer.span("AGENT", self.name, task=task[:70],
                              tools=",".join(self.registry.names())):
            return self.agent.solve(task)


class Team:
    """Coordinator. Assigns tasks, sequences roles, resolves the handoffs."""

    def __init__(self, registry: ToolRegistry, tracer: Tracer | None = None) -> None:
        self.tracer = tracer or Tracer()
        self.registry = registry
        self.researcher = Specialist("researcher", registry, self.tracer,
                                     ["rag_search", "web_search"])
        self.analyst = Specialist("analyst", registry, self.tracer,
                                  ["calculator", "weather"])

    def solve(self, goal: str) -> dict:
        started = time.perf_counter()
        board = Blackboard(goal, self.tracer)
        self.tracer.event("GOAL", goal, agent="team")

        # --- 1. Coordinator splits the goal by role ----------------------
        research_task, analysis_task = self._split(goal)
        with self.tracer.span("PLAN", "coordinator assigns roles",
                              researcher=bool(research_task), analyst=bool(analysis_task)):
            self.tracer.event("NOTE", f"researcher <- {research_task or 'nothing'}")
            self.tracer.event("NOTE", f"analyst    <- {analysis_task or 'nothing'}")

        # --- 2. Specialists work in parallel conceptually, sequentially here
        if research_task:
            result = self.researcher.work(research_task, board)
            for obs in result.observations:
                # Post the actual passage text, not the tool's score summary.
                # "3 passage(s), top score 0.24" is a debugging line, and a
                # writer that receives it can only paraphrase a paraphrase.
                # Handoffs must carry the evidence itself.
                for text, cites in Composer._render(obs):
                    board.post(Entry(
                        author="researcher", kind="evidence",
                        content=text, citations=cites, ok=obs.ok,
                    ))

        if analysis_task:
            result = self.analyst.work(analysis_task, board)
            for obs in result.observations:
                board.post(Entry(
                    author="analyst",
                    kind="number" if obs.tool == "calculator" else "measurement",
                    content=obs.line(),
                    citations=obs.citations,
                    ok=obs.ok,
                ))

        # --- 3. Critic reviews the board (no tools, just judgement) -------
        with self.tracer.span("REFLECT", "critic reviews blackboard") as ev:
            critique = self._critique(board)
            ev.detail["verdict"] = critique
            board.post(Entry(author="critic", kind="critique", content=critique))

        # --- 4. Writer composes from posted entries only ------------------
        with self.tracer.span("RESPOND", "writer composes"):
            answer = self._write(board, critique)
            board.post(Entry(author="writer", kind="answer", content=answer))

        return {
            "goal": goal,
            "answer": answer,
            "blackboard": board,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    # -- coordinator logic ------------------------------------------------

    ANALYSIS_HINTS = ("weather", "temperature", "calculate", "difference", "how much",
                      "compare", "compute", "sum", "average", "multiply", "warmer",
                      "colder", "humidity", "forecast")

    def _split(self, goal: str) -> tuple[str, str]:
        """Decide which parts of the goal belong to which specialist."""
        lowered = goal.lower()
        needs_analysis = any(h in lowered for h in self.ANALYSIS_HINTS)
        clauses = [c.strip() for c in goal.replace(";", ",").split(",") if c.strip()]

        if not needs_analysis:
            return goal, ""
        research_bits = [c for c in clauses
                         if not any(h in c.lower() for h in self.ANALYSIS_HINTS)]
        analysis_bits = [c for c in clauses
                         if any(h in c.lower() for h in self.ANALYSIS_HINTS)]
        return (
            ", ".join(research_bits),
            ", ".join(analysis_bits) or goal,
        )

    @staticmethod
    def _critique(board: Blackboard) -> str:
        findings = [e for e in board.entries if e.kind in ("evidence", "measurement", "number")]
        numbers = board.by_kind("number")
        failures = [e for e in board.entries if not e.ok]
        uncited = [e for e in findings if e.ok and not e.citations]

        problems = []
        if not findings:
            problems.append("no evidence was posted at all")
        if failures:
            problems.append(f"{len(failures)} posted item(s) came from failed steps")
        if uncited:
            problems.append(f"{len(uncited)} item(s) have no citation")
        if not problems:
            return (f"accepted: {len(findings)} finding(s), {len(numbers)} computed "
                    f"value(s), all cited")
        return "needs work: " + "; ".join(problems)

    @staticmethod
    def _write(board: Blackboard, critique: str) -> str:
        lines = [f"Goal: {board.goal}", ""]
        research = [e for e in board.entries if e.author == "researcher" and e.ok]
        analysis = [e for e in board.entries if e.author == "analyst" and e.ok]

        if research:
            lines.append("Research findings:")
            lines += [f"  - {e.content}" for e in research]
        if analysis:
            lines.append("")
            lines.append("Measurements and analysis:")
            lines += [f"  - {e.content}" for e in analysis]

        failed = [e for e in board.entries if not e.ok]
        if failed:
            lines.append("")
            lines.append("Unresolved:")
            lines += [f"  - {e.content}" for e in failed]

        cites = [c for e in board.entries for c in e.citations]
        if cites:
            lines.append("")
            lines.append("Sources: " + " ".join(dict.fromkeys(cites)))
        lines.append(f"Critic: {critique}")
        return "\n".join(lines)
