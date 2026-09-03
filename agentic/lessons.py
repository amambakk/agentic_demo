"""
Nine runnable lessons. Each one prints a narration, runs real code, and
ends with a takeaway. Run them with:

    python run_demo.py --lesson 3
    python run_demo.py --all
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from .agent import Agent
from .errors import ToolError
from .memory import EpisodicMemory
from .multi_agent import Team
from .planner import DecompositionPlanner, KeywordRouter, NaivePlanner
from .retrieval import VectorIndex
from .tools import build_default_registry
from .trace import Tracer, rule

DATA = Path(__file__).resolve().parents[1] / "data"


def _say(text: str) -> None:
    print(text)


def _para(text: str) -> None:
    """Print a wrapped narration paragraph."""
    print()
    print(textwrap.fill(" ".join(text.split()), width=78))
    print()


# ---------------------------------------------------------------------------
# Lesson 1 -- routing, and why the obvious version breaks
# ---------------------------------------------------------------------------


def lesson_01_routing() -> None:
    rule("LESSON 1  Routing: one query, one tool")
    _para("""
        The starting point is a keyword router: look for trigger words, return
        a tool name. It is the right first layer -- fast, free, deterministic,
        trivial to debug. It also breaks in three specific ways. Watch.
    """)

    probes = [
        "calculate 12 * 7",
        "Calculate 12 * 7",                      # capital C
        "Is there a newsletter about agents?",   # 'news' hides inside 'newsletter'
        "compare the weather in Lagos and Accra and then give the difference",
    ]

    print(f"{'query':<46} {'naive':<12} {'fixed':<12}")
    print("-" * 78)
    fixed = KeywordRouter()
    for q in probes:
        naive = NaivePlanner(q).which_tool_to_choose()
        chosen, _ = fixed.route(q)
        flag = "  <-- differs" if naive != chosen else ""
        shown = q if len(q) <= 45 else q[:42] + "..."
        print(f"{shown:<46} {naive:<12} {chosen:<12}{flag}")

    _para("""
        Bug 1, case sensitivity: 'Calculate' is not 'calculate', so the query
        silently falls through to the knowledge base. Nothing raises. The
        answer is just quietly wrong, which is the worst failure mode there is.
    """)
    _para("""
        Bug 2, substring matching: `word in query` is a substring test on a
        string, not a membership test on tokens. 'news' matches inside
        'newsletter'. The fix is to tokenise first.
    """)
    _para("""
        Bug 3, one tool only: the last query needs two weather lookups AND a
        calculation, in that order. A router cannot express that at all. That
        limitation is what forces the jump from routing to planning -- Lesson 5.
    """)
    print("TAKEAWAY: routing is planning with exactly one step. Normalise the")
    print("          input, match on tokens, and know when one step is not enough.")


# ---------------------------------------------------------------------------
# Lesson 2 -- tools are contracts
# ---------------------------------------------------------------------------


def lesson_02_tools() -> None:
    rule("LESSON 2  Tools are contracts, not functions")
    _para("""
        A tool is four things and only one of them is code: a name, a
        description, an argument schema, and a structured failure mode. The
        description is what a planner reads to decide when to call it -- so a
        vague description is a routing bug waiting to happen.
    """)
    registry = build_default_registry()
    print("The toolbox the planner is allowed to route to:\n")
    print(registry.catalog())

    _para("""
        Here is the `weather` tool serialised the way an LLM would actually see
        it for function calling. When people say 'the model picked the wrong
        tool', this JSON is usually the thing that was wrong.
    """)
    print(json.dumps(registry.get("weather").spec(), indent=2))

    _para("""
        And here is why failures must be structured. Three different problems,
        three different correct responses:
    """)
    calc, weather = registry.get("calculator"), registry.get("weather")
    for label, fn in [
        ("calculator with a divide by zero", lambda: calc.run(expression="10 / 0")),
        ("calculator with an injection attempt", lambda: calc.run(expression="__import__('os')")),
        ("weather for a city with no coverage", lambda: weather.run(city="Kano")),
    ]:
        try:
            fn()
        except ToolError as exc:
            print(f"  {label}")
            print(f"      kind={exc.kind:<14} retryable={exc.retryable}")
            print(f"      hint: {exc.hint}")

    print("\nTAKEAWAY: the error `kind` is what the executor branches on. Without")
    print("          it, every failure looks the same and recovery is guesswork.")


# ---------------------------------------------------------------------------
# Lesson 3 -- retrieval and corrective RAG
# ---------------------------------------------------------------------------


def lesson_03_retrieval() -> None:
    rule("LESSON 3  Memory as retrieval, and grading what you retrieve")
    index = VectorIndex.from_directory(DATA / "corpus")
    print(f"Knowledge base loaded: {index.stats()}\n")

    _para("""
        Retrieval is: vectorise the query, score every passage, take the top k.
        TF-IDF here, embeddings in production -- the pipeline shape is the same,
        only the quality of the scores changes.
    """)
    for hit in index.search("how does an agent recover from a tool failure", k=3):
        print(f"  {hit.score:.3f}  {hit.passage.citation:<28} {hit.passage.preview(90)}")

    _para("""
        Now the important part. Cosine similarity ALWAYS returns something.
        Ask the knowledge base about something it has never heard of and it
        will still hand back its three least-bad passages, with citations,
        looking exactly as trustworthy as a real answer:
    """)
    rag = build_default_registry().get("rag_search")
    for query in ["what is corrective rag", "tell me about jollof rice recipes"]:
        result = rag.run(query=query)
        print(f"  query: {query!r}")
        print(f"    -> {result.summary}")
        print(f"    -> grade: {result.grade.upper()}")

    print("\nTAKEAWAY: naive RAG answers from whatever came back. Corrective RAG")
    print("          grades it first, and a weak grade must change the plan --")
    print("          rewrite the query, fall back to another source, or say")
    print("          'my knowledge base does not cover this'.")

    _para("""
        The grade is only half of it. Something has to ACT on it. Watch the
        executor treat a weak grade as a soft failure and reach for the
        alternate source on its own:
    """)
    agent = Agent(build_default_registry(), Tracer())
    agent.solve("what do operators want most from their agents")
    _para("""
        Note the policy in that fallback: it only UPGRADES. If the alternate
        source had also come back weak, the original would have been kept and
        the weak grade left in place for the critic to see. Swapping weak
        evidence for different weak evidence hides the problem instead of
        fixing it.
    """)


# ---------------------------------------------------------------------------
# Lesson 4 -- the loop
# ---------------------------------------------------------------------------


def lesson_04_the_loop() -> None:
    rule("LESSON 4  The full loop, traced")
    _para("""
        GOAL -> PLAN -> ACT -> OBSERVE -> REFLECT -> RESPOND. Every line below
        is a trace event, indented by depth. This is the entire point of the
        project: an agent whose control flow you can read.
    """)
    agent = Agent(build_default_registry(), Tracer())
    result = agent.solve("explain what reflection means in an agent loop")
    print("\n--- ANSWER " + "-" * 66)
    print(result.answer)
    print("\nTrajectory metrics:", agent.tracer.summary())
    print("\nTAKEAWAY: the trace is the product. Answers without traces cannot")
    print("          be debugged, evaluated, or trusted.")


# ---------------------------------------------------------------------------
# Lesson 5 -- multi-step planning with dependencies
# ---------------------------------------------------------------------------


def lesson_05_planning() -> None:
    rule("LESSON 5  Decomposition: when one step is not enough")
    goal = "compare the weather in Lagos and Accra and give me the temperature difference"

    print(f"Goal: {goal}\n")
    print("A router says:      ", KeywordRouter().route(goal)[0], " (one tool, no order)")
    plan = DecompositionPlanner().plan(goal)
    print("A planner says:")
    for line in plan.describe():
        print("   ", line)
    print(f"\n    note: {plan.note}")

    _para("""
        Step 3 depends on steps 1 and 2. Its argument starts as a placeholder
        and has to be BOUND from the earlier observations at run time. Watch
        for the 'bound dependent arguments' event -- that binding step is
        where most multi-step agents actually break.
    """)
    agent = Agent(build_default_registry(), Tracer())
    result = agent.solve(goal)
    print("\n--- ANSWER " + "-" * 66)
    print(result.answer)
    print("\nTAKEAWAY: a plan is steps + arguments + ORDER + dependencies. Getting")
    print("          the tool right and the argument wrong is still a failure.")


# ---------------------------------------------------------------------------
# Lesson 6 -- failure and recovery
# ---------------------------------------------------------------------------


def lesson_06_recovery() -> None:
    rule("LESSON 6  Failure and recovery")
    _para("""
        Recovery code you have never watched execute does not work. So the
        weather tool here fails on a fixed schedule instead of at random.
        Three failures, three different correct responses.
    """)

    print(">>> CASE A: transient failure -> retry with backoff")
    print("    (weather is rigged to time out on its first call)\n")
    agent = Agent(build_default_registry(weather_fail_first=1), Tracer())
    agent.solve("what is the weather in Lagos")
    _para("""
        Look at the 'retrying' event. The user never saw the failure, and the
        answer is correct. That is what working recovery looks like -- which
        is also why you must test it deliberately: a run that silently
        recovered is indistinguishable from a run that never had a problem.
    """)

    print(">>> CASE B: not_found -> do NOT retry, fall back to another tool")
    print("    (the same arguments would fail identically, so retrying is waste)\n")
    agent = Agent(build_default_registry(), Tracer())
    result = agent.solve("what is the weather in Kano")
    _para("""
        Three things worth stopping on here. First, no retry: the executor saw
        kind=not_found and went straight to the fallback. Second, the fallback
        'succeeded' but answered a DIFFERENT question -- it returned a weather
        news article, not the weather in Kano. A successful fallback is not
        the same as a satisfied goal. Third, the critic caught exactly that
        ('evidence does not mention: kano'), tried one repair, got nowhere,
        and then stopped instead of spinning. That last brake is the progress
        check: if a replan produces no gap you have not already failed at,
        further loops are theatre.
    """)
    print("--- ANSWER " + "-" * 66)
    print(result.answer)

    print("\n>>> CASE C: unavailable -> route around it and remember the outage")
    registry = build_default_registry()
    registry.get("weather").offline = True
    agent = Agent(registry, Tracer())
    result = agent.solve("what is the weather in Lagos and in Accra")
    print("\n--- ANSWER " + "-" * 66)
    print(result.answer)

    print("\nTAKEAWAY: the failure `kind` picks the strategy. transient -> retry;")
    print("          invalid_input -> repair the args; not_found -> fall back;")
    print("          unavailable -> mark the tool dead for the rest of the run.")


# ---------------------------------------------------------------------------
# Lesson 7 -- reflection driving a replan
# ---------------------------------------------------------------------------


def lesson_07_reflection() -> None:
    rule("LESSON 7  Reflection that actually changes behaviour")
    _para("""
        Reflection is a checklist, not a mood. The critic asks: did every step
        run, did anything fail, was any retrieval weak, does the evidence
        mention every content word in the goal? Each failed check becomes a
        'gap' -- phrased so it can be fed straight back to the PLANNER as a
        new sub-goal. That is the entire replanning mechanism.
    """)
    agent = Agent(build_default_registry(), Tracer())
    result = agent.solve("what does the knowledge base say about quantum tunnelling in agents")

    _para("""
        Follow what just happened. The knowledge base returned weakly relevant
        passages, so the executor tried the alternate source and got a
        confident-looking set of news articles. That is the dangerous moment:
        the run now LOOKS successful. The critic is what catches it -- the
        evidence never mentions 'quantum' or 'tunnelling', so the words the
        user actually asked about went unanswered. One repair attempt, no
        progress, stop. The answer says so plainly instead of dressing up
        three irrelevant articles as a response.
    """)

    print("\n--- ANSWER " + "-" * 66)
    print(result.answer)
    print(f"\nIterations used: {result.iterations} (budget {agent.max_iterations})")
    if result.reflection:
        print("Critic verdict :", result.reflection.describe())
        for issue in result.reflection.issues:
            print("   issue:", issue)

    print("\nTAKEAWAY: bounded reflection. An agent that loops forever never")
    print("          answers; one that never loops never notices it was wrong.")


# ---------------------------------------------------------------------------
# Lesson 8 -- multi-agent
# ---------------------------------------------------------------------------


def lesson_08_multi_agent() -> None:
    rule("LESSON 8  Multi-agent collaboration")
    _para("""
        Four roles, coordinated over a shared blackboard. The researcher has
        only retrieval tools; the analyst has only calculator and weather; the
        critic has no tools at all; the writer composes from posted entries
        only. Specialisation is enforced by the TOOLBOX, not by instructions --
        the researcher cannot do arithmetic because it does not have a
        calculator, which is far more reliable than asking it not to.
    """)
    team = Team(build_default_registry(), Tracer())
    out = team.solve(
        "explain how multi agent systems coordinate, and compare the weather in Lagos and Accra"
    )
    print("\n--- ANSWER " + "-" * 66)
    print(out["answer"])
    print(f"\nWall time: {out['duration_ms']} ms across {len(out['blackboard'].entries)} board entries")

    print("\nTAKEAWAY: more agents is not automatically better. You pay in calls,")
    print("          latency and coordination bugs. The payoff is role clarity")
    print("          and an independent review step -- not headcount.")


# ---------------------------------------------------------------------------
# Lesson 9 -- memory across runs, and evaluation
# ---------------------------------------------------------------------------


def lesson_09_memory_and_eval() -> None:
    rule("LESSON 9  Episodic memory and evaluating a trajectory")
    store = Path("./.agent_memory/episodes.json")
    memory = EpisodicMemory(store)
    memory.clear()
    memory = EpisodicMemory(store)

    print("Run 1 (cold memory):")
    agent = Agent(build_default_registry(), Tracer(enabled=False), episodic=memory)
    r1 = agent.solve("explain the role of the critic in an agent loop")
    print(f"   reused_memory={r1.reused_memory}  iterations={r1.iterations}  {r1.duration_ms} ms")

    print("\nRun 2 (same goal, warm memory) -- watch the MEMORY event:")
    agent = Agent(build_default_registry(), Tracer(), episodic=EpisodicMemory(store))
    r2 = agent.solve("explain the role of the critic in an agent loop")
    print(f"\n   reused_memory={r2.reused_memory}")
    print(f"   episodic stats: {EpisodicMemory(store).stats()}")
    print(f"   persisted at: {store.resolve()}")

    _para("""
        Now evaluation. An agent's output is a TRAJECTORY, not a string. Two
        runs can produce the same answer via completely different paths, and
        only one of them is reliable. So the metrics that matter come off the
        trace:
    """)
    agent = Agent(build_default_registry(weather_fail_first=1), Tracer(enabled=False))
    agent.solve("compare the weather in Lagos and Accra and give the difference")
    metrics = agent.tracer.summary()
    for key, value in metrics.items():
        print(f"   {key:<18} {value}")
    print("\n   tool_selection = ", agent.executor.registry.names())

    print("\nTAKEAWAY: score the path, not just the destination. Step count,")
    print("          tool-selection accuracy, retry count, recovery success and")
    print("          tail latency are the numbers that predict production pain.")


LESSONS = [
    ("Routing and its three classic bugs", lesson_01_routing),
    ("Tools as contracts with structured failures", lesson_02_tools),
    ("Retrieval, and corrective RAG grading", lesson_03_retrieval),
    ("The full observable agent loop", lesson_04_the_loop),
    ("Decomposition planning with dependencies", lesson_05_planning),
    ("Failure recovery: retry, fallback, route around", lesson_06_recovery),
    ("Reflection that triggers a replan", lesson_07_reflection),
    ("Multi-agent collaboration on a blackboard", lesson_08_multi_agent),
    ("Episodic memory and trajectory evaluation", lesson_09_memory_and_eval),
]


def run_lesson(n: int) -> None:
    if not 1 <= n <= len(LESSONS):
        raise SystemExit(f"lesson must be 1..{len(LESSONS)}")
    LESSONS[n - 1][1]()


def run_all() -> None:
    for i, (title, fn) in enumerate(LESSONS, 1):
        fn()
        print("\n")


def list_lessons() -> None:
    rule("LESSONS")
    for i, (title, _) in enumerate(LESSONS, 1):
        print(f"  {i}. {title}")
    print("\nRun one with:  python run_demo.py --lesson 4")
    print("Run them all:  python run_demo.py --all")
