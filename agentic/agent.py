"""
The agent loop.

    GOAL -> PLAN -> ACT -> OBSERVE -> REFLECT -> (loop or) RESPOND

Everything the agent does passes through the tracer, so a run is fully
auditable after the fact. Read `Agent.solve` top to bottom: it is short on
purpose, because the interesting work has been pushed into the planner,
the executor and the critic, each of which can be tested alone.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .executor import Executor, Observation
from .memory import Episode, EpisodicMemory, Scratchpad
from .planner import DecompositionPlanner, Plan, Step
from .reflection import Critic, Reflection
from .tools import ToolRegistry
from .trace import Tracer


def _norm(text: str) -> str:
    """Normalise a gap string so re-phrasings of the same gap collide."""
    return " ".join(sorted(re.findall(r"[a-z0-9]+", text.lower())))


@dataclass
class AgentResult:
    goal: str
    answer: str
    plan: Plan | None
    observations: list[Observation] = field(default_factory=list)
    reflection: Reflection | None = None
    iterations: int = 0
    duration_ms: float = 0.0
    reused_memory: bool = False

    @property
    def citations(self) -> list[str]:
        seen, out = set(), []
        for obs in self.observations:
            for c in obs.citations:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out


class Agent:
    def __init__(
        self,
        registry: ToolRegistry,
        tracer: Tracer | None = None,
        planner: DecompositionPlanner | None = None,
        critic: Critic | None = None,
        episodic: EpisodicMemory | None = None,
        name: str = "agent",
        max_iterations: int = 3,
    ) -> None:
        self.registry = registry
        self.tracer = tracer or Tracer()
        self.planner = planner or DecompositionPlanner()
        self.critic = critic or Critic()
        self.episodic = episodic or EpisodicMemory()
        self.executor = Executor(registry, self.tracer)
        self.name = name
        self.max_iterations = max_iterations

    # -----------------------------------------------------------------

    def solve(self, goal: str) -> AgentResult:
        started = time.perf_counter()
        scratch = Scratchpad(goal=goal)
        observations: list[Observation] = []
        reflection: Reflection | None = None
        plan: Plan | None = None
        gaps: list[str] = []
        attempted: set[str] = set()   # gaps already tried -- see the progress check
        reused = False

        self.tracer.event("GOAL", goal, agent=self.name)

        # --- memory check: have we solved this before? -------------------
        prior = self.episodic.recall_similar(goal)
        if prior:
            reused = True
            self.tracer.event(
                "MEMORY", "similar past episode found",
                past_goal=prior.goal, tools=",".join(prior.tools_used),
            )

        for iteration in range(1, self.max_iterations + 1):
            # --- PLAN ----------------------------------------------------
            with self.tracer.span("PLAN", f"iteration {iteration}", gaps=len(gaps)) as ev:
                plan = self.planner.plan(goal, gaps=gaps or None)
                ev.detail["steps"] = len(plan)
                for line in plan.describe():
                    self.tracer.event("NOTE", line)
                if plan.note:
                    self.tracer.event("NOTE", plan.note)

            # --- ACT + OBSERVE -------------------------------------------
            for step in plan.steps:
                self._resolve_dependencies(step, observations)
                obs = self.executor.run_step(step)
                observations.append(obs)
                self._write_facts(obs, scratch)

            # --- REFLECT --------------------------------------------------
            with self.tracer.span("REFLECT", f"iteration {iteration}") as ev:
                reflection = self.critic.review(goal, plan, observations)
                ev.detail["verdict"] = reflection.describe()
                for issue in reflection.issues:
                    self.tracer.event("NOTE", f"issue: {issue}")

            if reflection.complete:
                break
            if iteration == self.max_iterations:
                self.tracer.event(
                    "NOTE", "iteration budget exhausted; answering with what we have"
                )
                break

            # Progress check. An agent that keeps re-attempting a gap it has
            # already failed is not reasoning, it is spinning -- and spinning
            # looks exactly like working until you read the trace. Two
            # independent brakes: a hard iteration budget, and this one.
            fresh = [g for g in reflection.gaps if _norm(g) not in attempted]
            if not fresh:
                self.tracer.event(
                    "NOTE", "no new gaps to pursue; stopping and answering honestly",
                    repeated=len(reflection.gaps),
                )
                break
            attempted.update(_norm(g) for g in fresh)
            gaps = fresh
            self.tracer.event("NOTE", "replanning", gaps="; ".join(gaps))

        # --- RESPOND ------------------------------------------------------
        answer = Composer().compose(goal, observations, reflection)
        self.tracer.event("RESPOND", "answer composed", chars=len(answer))

        duration = round((time.perf_counter() - started) * 1000, 2)
        episode = Episode(
            goal=goal,
            plan=plan.describe() if plan else [],
            tools_used=sorted({o.tool for o in observations}),
            success=bool(reflection and reflection.complete),
            gaps=list(reflection.gaps) if reflection else [],
            answer=answer,
            duration_ms=duration,
        )
        self.episodic.add(episode)
        self.tracer.event("MEMORY", "episode stored", success=episode.success)

        return AgentResult(
            goal=goal,
            answer=answer,
            plan=plan,
            observations=observations,
            reflection=reflection,
            iterations=min(iteration, self.max_iterations),
            duration_ms=duration,
            reused_memory=reused,
        )

    # -----------------------------------------------------------------
    # Dependency resolution: binding one step's output into the next
    # step's arguments. This is the part people skip, and it is exactly
    # where multi-step agents break.
    # -----------------------------------------------------------------

    def _resolve_dependencies(self, step: Step, observations: list[Observation]) -> None:
        if not step.depends_on:
            return
        sources = [o for o in observations if o.step.id in step.depends_on and o.ok]
        if step.tool == "calculator" and "<" in str(step.args.get("expression", "")):
            temps = [
                o.data.get("temp_c")
                for o in sources
                if isinstance(o.data, dict) and o.data.get("temp_c") is not None
            ]
            if len(temps) >= 2:
                step.args["expression"] = f"{temps[0]} - {temps[1]}"
                self.tracer.event(
                    "NOTE", "bound dependent arguments",
                    step=step.id, expression=step.args["expression"],
                )
            else:
                # Inputs never arrived. Do not call the tool with a
                # placeholder; mark the step so the critic sees it.
                step.args["expression"] = ""
                self.tracer.event(
                    "NOTE", "dependency unresolved", step=step.id,
                    available=len(temps),
                )

    @staticmethod
    def _write_facts(obs: Observation, scratch: Scratchpad) -> None:
        if not obs.ok or not isinstance(obs.data, dict):
            return
        if "temp_c" in obs.data:
            scratch.remember_fact(
                f"temp_c:{obs.data.get('city')}", obs.data["temp_c"], obs.citations[0] if obs.citations else ""
            )
        if "value" in obs.data:
            scratch.remember_fact("calc_result", obs.data["value"], "[calculator]")


# ---------------------------------------------------------------------------
# Response composition
# ---------------------------------------------------------------------------


class Composer:
    """Turn observations into an answer.

    In an LLM system this is the one place a model would generate prose.
    Doing it with templates here is a feature, not a limitation: it makes
    it obvious that every sentence in the answer is traceable to an
    observation, which is what "grounded" actually means.
    """

    def compose(
        self, goal: str, observations: list[Observation], reflection: Reflection | None
    ) -> str:
        lines: list[str] = []
        cites: list[str] = []
        good = [o for o in observations if o.ok]
        bad = [o for o in observations if not o.ok]

        if not good:
            lines.append("I could not gather any usable evidence for this request.")
        else:
            lines.append(f"Goal: {goal}")
            lines.append("")
            lines.append("What I found:")
            # A replan often re-retrieves a passage the first pass already
            # found. Printing it twice makes the answer look like it has more
            # support than it does, so evidence is deduplicated on render.
            seen: set[str] = set()
            for obs in good:
                for text, used in self._render(obs):
                    key = text[:120]
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(f"  - {text}")
                    cites.extend(used)

        if bad:
            lines.append("")
            lines.append("What I could not resolve:")
            for obs in bad:
                msg = (obs.error or {}).get("message", "unknown failure")
                hint = (obs.error or {}).get("hint")
                lines.append(f"  - {obs.step.intent}: {msg}" + (f" ({hint})" if hint else ""))

        if cites:
            lines.append("")
            lines.append("Sources: " + " ".join(dict.fromkeys(cites)))

        if reflection:
            lines.append(f"Confidence: {reflection.confidence} (score {reflection.score:.2f})")
            if reflection.issues:
                lines.append("Caveats: " + "; ".join(reflection.issues[:3]))

        return "\n".join(lines)

    @staticmethod
    def _render(obs: Observation, top_n: int = 2) -> list[tuple[str, list[str]]]:
        """Return (text, citations) pairs -- cite only what is actually shown.

        Listing every retrieved citation while quoting one passage is a small
        dishonesty that shows up constantly in RAG systems: the answer looks
        three times better sourced than it is.
        """
        if obs.tool == "rag_search" and isinstance(obs.data, dict):
            return [
                (f"{p['preview']} {p['citation']}", [p["citation"]])
                for p in obs.data["passages"][:top_n]
            ]
        if obs.tool == "web_search" and isinstance(obs.data, dict):
            results = obs.data["results"][:top_n]
            return [
                (text, [cite])
                for text, cite in zip(results, obs.citations[: len(results)])
            ]
        return [(obs.line(), obs.citations)]
