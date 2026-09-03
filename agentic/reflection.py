"""
Reflection: mechanical self-critique.

Teaching point
--------------
"Reflection" sounds mystical; in a working system it is a checklist. The
critic below asks five questions that can each be answered by looking at
the trace:

    1. did every planned step produce an observation?
    2. did any step fail outright?
    3. was any retrieval graded weak?
    4. does the evidence mention every content word from the goal?
    5. did a dependent step run without its inputs?

Each failed check becomes a *gap string*, and a gap string is phrased so it
can be fed straight back to the planner as a new sub-goal. That is the
whole replanning mechanism -- no magic, just a critic that writes its
complaints in the planner's input language.

Note also that the loop is bounded. An agent that reflects forever never
answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .executor import Observation
from .planner import ROUTING_RULES, Plan
from .retrieval import stem, tokenize

# Words that describe the *task* rather than its subject matter.
#
# Teaching point -- why this set has to exist
# -------------------------------------------
# The coverage check asks "does the evidence mention what the goal was
# about?". But a goal like "compare the weather in Lagos and Accra and give
# the temperature difference" contains words that will NEVER appear in the
# evidence: a weather reading says "31C, humid", not "compare" or
# "temperature difference". Scoring those words as missing makes the critic
# report a failure on a run that worked perfectly -- and a critic that cries
# wolf gets switched off, which is worse than having no critic.
#
# Two groups are excluded. Imperative verbs, because they are instructions
# to the agent rather than facts to be found. And every router trigger word,
# because a trigger that selected a tool which then SUCCEEDED has already
# been accounted for -- the tool running is the evidence that the term was
# handled.
_IMPERATIVES = {
    "compare", "comparison", "give", "tell", "show", "explain", "describe",
    "find", "get", "list", "say", "mean", "means", "want", "need", "please",
    "question", "answer", "versus", "vs", "about", "knowledge", "base",
}
TASK_TERMS: set[str] = {stem(w) for w in _IMPERATIVES} | {
    stem(trigger)
    for _tool, triggers in ROUTING_RULES
    for trigger in triggers
    if " " not in trigger
}


@dataclass
class Reflection:
    complete: bool
    score: float                       # 0..1, coverage-weighted
    issues: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)   # feed these back to the planner
    confidence: str = "medium"

    def describe(self) -> str:
        verdict = "goal satisfied" if self.complete else "gaps remain"
        return f"{verdict} (score {self.score:.2f}, {len(self.issues)} issue(s))"


class Critic:
    def __init__(self, coverage_threshold: float = 0.6) -> None:
        self.coverage_threshold = coverage_threshold

    def review(self, goal: str, plan: Plan, observations: list[Observation]) -> Reflection:
        issues: list[str] = []
        gaps: list[str] = []

        # 1 + 2: execution completeness
        done_ids = {o.step.id for o in observations}
        for step in plan.steps:
            if step.id not in done_ids:
                issues.append(f"step #{step.id} never ran ({step.intent})")
                gaps.append(step.intent)

        for obs in observations:
            if not obs.ok:
                hint = (obs.error or {}).get("hint")
                issues.append(
                    f"{obs.tool} failed: {obs.error and obs.error.get('message')}"
                    + (f" -- {hint}" if hint else "")
                )
                # The gap is the *sub-goal*, not the error text. Gaps are fed
                # back to the planner as input, so appending diagnostics here
                # produces garbage queries on the next iteration.
                gaps.append(obs.step.intent)

            # 3: weak retrieval is a soft failure, and must be treated as one
            elif obs.grade == "weak":
                issues.append(f"{obs.tool} returned weakly relevant evidence")
                # A gap is a REQUEST FOR ACTION, so only raise one if an
                # action remains. If the executor already tried every
                # alternate source and they were no better, re-planning would
                # just re-run the tool that already failed. Recording the
                # caveat and stopping is the honest move.
                if not obs.alternates_tried and obs.recovered_with is None:
                    gaps.append(f"find better sources for: {obs.step.intent}")

            # 5: a dependent step that ran on placeholder arguments
            if obs.ok and obs.step.depends_on:
                arg_text = " ".join(str(v) for v in obs.step.args.values())
                if "<" in arg_text:
                    issues.append(f"step #{obs.step.id} ran with unresolved inputs")

        # 4: keyword coverage of the goal by the accumulated evidence
        coverage = self._coverage(goal, observations)
        if coverage < self.coverage_threshold and not gaps:
            missing = self._missing_terms(goal, observations)
            if missing:
                issues.append(f"evidence does not mention: {', '.join(missing)}")
                gaps.append(f"what about {' '.join(missing)}")

        usable = [o for o in observations if o.usable]
        score = round(coverage * (len(usable) / max(len(observations), 1)), 3)
        complete = not gaps and bool(usable)

        return Reflection(
            complete=complete,
            score=score,
            issues=issues,
            gaps=gaps[:3],  # bounded: do not spawn an unbounded replan
            confidence="high" if score >= 0.75 else "medium" if score >= 0.4 else "low",
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _evidence_tokens(observations: list[Observation]) -> set[str]:
        """Tokens the run can legitimately claim to have addressed.

        Subtlety worth understanding: a successful `weather` call answers the
        word "weather" even though the string "weather" never appears in
        "Lagos: 31C, humid". So a term that the router consumed, on a step
        that then succeeded, counts as covered. Without this the critic
        reports a false gap on almost every successful run -- and a critic
        that cries wolf gets ignored, which is worse than having none.
        """
        parts: list[str] = []
        for obs in observations:
            if not obs.ok:
                continue
            parts.append(obs.summary or "")
            parts.append(str(obs.data))
            parts.append(obs.tool.replace("_", " "))          # 'web search'
            match = re.search(r"'([^']+)'", obs.step.reason or "")
            if match:
                parts.append(match.group(1))                  # the matched trigger
        return set(tokenize(" ".join(parts)))

    def _coverage(self, goal: str, observations: list[Observation]) -> float:
        goal_terms = self._content_terms(goal)
        if not goal_terms:
            return 1.0 if any(o.usable for o in observations) else 0.0
        if not observations:
            return 0.0
        evidence = self._evidence_tokens(observations)
        return round(len(goal_terms & evidence) / len(goal_terms), 3)

    @staticmethod
    def _content_terms(goal: str) -> set[str]:
        """The parts of the goal that name subject matter, not instructions."""
        return {t for t in tokenize(goal) if t not in TASK_TERMS and len(t) > 2}

    def _missing_terms(self, goal: str, observations: list[Observation]) -> list[str]:
        """Goal terms with no trace in the evidence, in their original form.

        Matching happens on stems, but the words are reported as the user
        wrote them. Gap strings are fed back to the planner and shown to the
        user, and 'quantum tunnell' reads like a bug even when the matching
        underneath it was correct.
        """
        evidence = self._evidence_tokens(observations)
        seen, out = set(), []
        for surface in tokenize(goal, apply_stem=False):
            root = stem(surface)
            if root in evidence or root in TASK_TERMS or len(surface) <= 3:
                continue
            if root in seen:
                continue
            seen.add(root)
            out.append(surface)
        return out[:4]
