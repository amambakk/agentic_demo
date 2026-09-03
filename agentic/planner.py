"""
Planning, in three escalating versions.

    1. NaivePlanner       the base case, with three bugs that are shipped to production constantly
    2. KeywordRouter      the same idea, with the bugs fixed
    3. DecompositionPlanner  goal -> ordered, multi-step, dependency-aware plan

Reading these in order is the whole lesson: routing is a special case of
planning where the plan happens to have exactly one step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieval import stem, tokenize
from .tools.weather import WEATHER_DB


# ---------------------------------------------------------------------------
# 1. The starting point
# ---------------------------------------------------------------------------


class NaivePlanner:
    """The original router, kept exactly as written so we can break it.

    Three real bugs, all of which ship to production constantly:

    a) CASE SENSITIVITY. "Calculate 2+2" contains "Calculate", not
       "calculate", so the first branch misses and the query silently falls
       through to rag_search. Silent misrouting is the worst kind: nothing
       errors, the answer is just quietly wrong.

    b) SUBSTRING MATCHING. `word in self.input_query` is a substring test on
       a string, not a membership test on tokens. "Is there a newsletter?"
       matches "news". "I need a plussize chart" matches "plus".

    c) SINGLE TOOL, NO ORDER. It returns one tool name. A goal needing a
       lookup *and* a calculation cannot be expressed at all.
    """

    def __init__(self, input_query: str) -> None:
        self.input_query = input_query

    def which_tool_to_choose(self) -> str:
        if any(word in self.input_query for word in ["calculate", "multiply", "plus", "minus"]):
            return "calculator"
        elif any(word in self.input_query for word in ["weather", "temperature"]):
            return "weather"
        elif any(word in self.input_query for word in ["latest", "today", "news"]):
            return "web_search"
        else:
            return "rag_search"


# ---------------------------------------------------------------------------
# 2. The same idea, fixed
# ---------------------------------------------------------------------------

ROUTING_RULES: list[tuple[str, tuple[str, ...]]] = [
    # (tool, trigger tokens) -- order matters, first match wins
    ("calculator", ("calculate", "compute", "multiply", "plus", "minus", "times",
                    "divided", "difference", "sum", "average", "percent", "how much")),
    ("weather",    ("weather", "temperature", "forecast", "humidity", "hot", "cold", "raining")),
    ("web_search", ("latest", "today", "news", "recent", "current", "this week",
                    "yesterday", "announced")),
    ("rag_search", ()),  # default
]


class KeywordRouter:
    """Case-insensitive, token-aware, and it explains itself."""

    def __init__(self, rules: list[tuple[str, tuple[str, ...]]] | None = None) -> None:
        self.rules = rules or ROUTING_RULES

    def route(self, query: str) -> tuple[str, str]:
        """Return (tool_name, reason).

        Both sides are normalised the same way -- lowercased, tokenised and
        stemmed -- so 'Calculate', 'calculating' and 'calculation' all reach
        the same rule. Normalising only the query is a classic half-fix that
        leaves inflected forms unrouted.
        """
        lowered = query.lower()
        tokens = set(tokenize(lowered, drop_stopwords=False))
        for tool, triggers in self.rules:
            if not triggers:
                continue
            for trigger in triggers:
                if " " in trigger:                      # phrase: substring check
                    hit = trigger in lowered
                else:
                    hit = stem(trigger) in tokens
                if hit:
                    return tool, f"matched trigger {trigger!r}"
        return "rag_search", "no trigger matched; defaulting to the knowledge base"


# ---------------------------------------------------------------------------
# 3. Real planning: decomposition
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One unit of work: a tool, its arguments, and why we are doing it."""

    intent: str                       # the sub-goal in words
    tool: str
    args: dict = field(default_factory=dict)
    reason: str = ""
    depends_on: list[int] = field(default_factory=list)
    id: int = 0

    def describe(self) -> str:
        arg_text = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        dep = f" after#{self.depends_on}" if self.depends_on else ""
        return f"#{self.id} {self.tool}({arg_text}){dep} :: {self.intent}"


@dataclass
class Plan:
    goal: str
    steps: list[Step] = field(default_factory=list)
    note: str = ""

    def add(self, step: Step) -> Step:
        step.id = len(self.steps) + 1
        self.steps.append(step)
        return step

    def tools(self) -> list[str]:
        return [s.tool for s in self.steps]

    def describe(self) -> list[str]:
        return [s.describe() for s in self.steps]

    def __len__(self) -> int:
        return len(self.steps)


# Split a compound request into clauses. Deliberately simple and inspectable.
CLAUSE_SPLIT = re.compile(
    r"\s*(?:,\s*(?:and\s+)?then\s+|\s+and\s+then\s+|\s*;\s*|\s*\?\s*|\s+and\s+also\s+)",
    re.IGNORECASE,
)
CITY_RE = re.compile(r"\b(?:in|for|at)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)")
EXPR_RE = re.compile(r"[-+]?\d[\d\s.,]*(?:\s*[-+*/^x]\s*[-+]?\d[\d\s.,]*)+")


class DecompositionPlanner:
    """Split the goal into sub-goals, route each one, order the result.

    Still no LLM. Every decision is a rule you can read, which is exactly
    what makes it a good teaching artefact -- when the plan is wrong you can
    point at the line that produced it.
    """

    COMPARISON_WORDS = ("compare", "difference", "versus", " vs ", "warmer", "colder",
                        "hotter", "cooler", "between")

    def __init__(self, router: KeywordRouter | None = None) -> None:
        self.router = router or KeywordRouter()

    def plan(self, goal: str, gaps: list[str] | None = None) -> Plan:
        plan = Plan(goal=goal)

        # Replanning path: the critic told us exactly what is missing.
        if gaps:
            plan.note = "replan driven by critic gaps"
            for gap in gaps:
                tool, reason = self.router.route(gap)
                plan.add(Step(intent=gap, tool=tool,
                              args=self._extract_args(tool, gap, goal),
                              reason=f"gap: {reason}"))
            return plan

        clauses = [c.strip() for c in CLAUSE_SPLIT.split(goal) if c and c.strip()]
        cities = self._cities_in(goal)

        # Special case worth teaching: a comparison of two named cities is
        # three steps with a real dependency, not one step.
        if len(cities) >= 2 and self._is_comparison(goal):
            a, b = cities[0], cities[1]
            s1 = plan.add(Step(intent=f"look up weather in {a}", tool="weather",
                               args={"city": a}, reason="comparison needs both operands"))
            s2 = plan.add(Step(intent=f"look up weather in {b}", tool="weather",
                               args={"city": b}, reason="comparison needs both operands"))
            plan.add(Step(intent=f"compute the temperature difference between {a} and {b}",
                          tool="calculator", args={"expression": "<from steps 1 and 2>"},
                          reason="arithmetic must come after the lookups",
                          depends_on=[s1.id, s2.id]))
            plan.note = "comparison detected: 2 lookups then 1 dependent calculation"
            return plan

        for clause in clauses:
            tool, reason = self.router.route(clause)
            # One clause can still need two tools: "weather in Lagos and Accra"
            if tool == "weather":
                clause_cities = self._cities_in(clause) or cities
                if len(clause_cities) > 1:
                    for city in clause_cities:
                        plan.add(Step(intent=f"look up weather in {city}", tool="weather",
                                      args={"city": city},
                                      reason="one call per city; the tool takes one city"))
                    continue
            plan.add(Step(intent=clause, tool=tool,
                          args=self._extract_args(tool, clause, goal), reason=reason))

        if not plan.steps:  # empty or unparseable goal
            plan.add(Step(intent=goal, tool="rag_search", args={"query": goal},
                          reason="fallback: nothing parseable in the goal"))
        return plan

    # -- argument extraction ---------------------------------------------

    def _extract_args(self, tool: str, clause: str, goal: str) -> dict:
        if tool == "weather":
            cities = self._cities_in(clause) or self._cities_in(goal)
            return {"city": cities[0] if cities else ""}
        if tool == "calculator":
            match = EXPR_RE.search(clause)
            return {"expression": match.group(0).strip() if match else ""}
        return {"query": clause}

    @staticmethod
    def _cities_in(text: str) -> list[str]:
        """Prefer known cities; fall back to a capitalised noun after in/for/at."""
        found: list[str] = []
        lowered = text.lower()
        for city in WEATHER_DB:
            idx = lowered.find(city)
            if idx >= 0:
                found.append((idx, city.title()))
        if found:
            found.sort()
            return [c for _, c in found]
        return [m.strip() for m in CITY_RE.findall(text)]

    def _is_comparison(self, goal: str) -> bool:
        lowered = f" {goal.lower()} "
        return any(word in lowered for word in self.COMPARISON_WORDS)
