# Agentic AI, Offline

A complete, runnable teaching implementation of an agentic AI system.

**No LLM. No network. No API key. No third-party packages.** Python 3.10+ and the
standard library, nothing else.

Every decision a language model would normally make — which tool to call, how to
decompose a goal, whether the evidence is good enough, whether to try again — is
made here by a rule you can open and read. That is the point. The architecture
is the thing worth learning, and in a real system the architecture is invisible,
buried behind a prompt and a model you cannot inspect. Here you can put a
breakpoint on the agent's judgement.

```
GOAL  ->  PLAN  ->  ACT  ->  OBSERVE  ->  REFLECT  ->  RESPOND
            ^                                |
            +--------- replan on gaps -------+
```

## Start here

```bash
cd agentic_demo
python run_demo.py                 # guided tour
python run_demo.py --lesson 1      # lessons 1-9, in order
python run_demo.py --all           # every lesson end to end
python run_demo.py --test          # 36 unit tests
```

Then start poking at it:

```bash
python run_demo.py --ask "compare the weather in Lagos and Accra and give the difference"
python run_demo.py --team "explain multi agent coordination, and the weather in Tokyo"
python run_demo.py --ask "what is the weather in Kano" --fail-weather 1
python run_demo.py --repl
python run_demo.py --tools                       # the tool catalog the planner sees
python run_demo.py --ask "what is corrective rag" --trace-json trace.json
```

## The nine lessons

| # | Topic | The thing to actually notice |
|---|-------|------------------------------|
| 1 | Routing | Three real bugs in the obvious keyword router, run side by side |
| 2 | Tools as contracts | The description drives routing; the error `kind` drives recovery |
| 3 | Retrieval + corrective RAG | Cosine similarity always returns something. Grade it |
| 4 | The full loop | Every stage, traced, with trajectory metrics |
| 5 | Decomposition planning | Step 3 depends on steps 1 and 2, and must be *bound* at run time |
| 6 | Failure and recovery | Retry / repair / fall back / route around — one per failure kind |
| 7 | Reflection | A critic that changes behaviour, with two brakes on the loop |
| 8 | Multi-agent | Specialisation enforced by the toolbox, not by instructions |
| 9 | Memory and evaluation | Episodes across runs; scoring the path, not just the answer |

Lesson 1 starts from a keyword router — the kind almost everyone writes first —
and runs it against a fixed version on the same queries:

```
query                                          naive        fixed
------------------------------------------------------------------------------
calculate 12 * 7                               calculator   calculator
Calculate 12 * 7                               rag_search   calculator    <-- differs
Is there a newsletter about agents?            web_search   rag_search    <-- differs
compare the weather in Lagos and Accra and...  weather      calculator    <-- differs
```

A capital letter silently reroutes the query to the wrong tool. `"news"` matches
inside `"newsletter"`. And the last query needs two lookups plus a calculation in
that order, which a router cannot express at all — which is what forces the jump
from routing to planning.

## Layout

```
agentic_demo/
├── run_demo.py              CLI entry point
├── agentic/
│   ├── errors.py            ToolError: transient | invalid_input | not_found | unavailable
│   ├── trace.py             nested timestamped event log -> console tree or JSON
│   ├── retrieval.py         tokenizer, stemmer, TF-IDF, cosine ranking
│   ├── memory.py            Scratchpad (run) / EpisodicMemory (disk) / semantic index
│   ├── planner.py           NaivePlanner -> KeywordRouter -> DecompositionPlanner
│   ├── executor.py          retry with backoff, fallback chains, dead-tool tracking
│   ├── reflection.py        Critic: five checks -> gaps fed back to the planner
│   ├── agent.py             the loop, dependency binding, answer composition
│   ├── multi_agent.py       Blackboard + Researcher/Analyst/Critic/Writer + Team
│   ├── lessons.py           the nine narrated lessons
│   └── tools/               calculator, weather, web_search, rag_search, notepad
├── data/corpus/             9 documents — the knowledge base for RAG
├── data/web/                6 dated articles — the simulated "internet"
└── tests/test_agentic.py    36 tests
```

## Ideas the code is built to make visible

**Routing is planning with one step.** Same operation, different arity. Once you
see it that way the "should I use a router or a planner" question answers itself.

**A tool is a contract, not a function.** Name, description, argument schema,
failure mode. `Tool.spec()` emits the JSON an LLM would actually be shown for
function calling — when a model "picks the wrong tool", that JSON is usually
what was wrong.

**Failures have kinds, and the kind picks the strategy.** Retrying a
`not_found` with identical arguments produces an identical `not_found`. The
executor branches on the classification:

```
transient      -> retry with backoff, bounded
invalid_input  -> do not retry; repair the arguments
not_found      -> do not retry; fall back to a broader source
unavailable    -> route around it, and remember the outage for the rest of the run
```

**Corrective RAG.** Cosine similarity never returns nothing useful — it returns
its least-bad guesses, with citations, looking exactly as trustworthy as a real
answer. So retrieval grades itself, and a weak grade makes the executor try the
alternate source. The fallback only *upgrades*: if the alternate is also weak,
the original is kept and the weak grade stays visible to the critic. Swapping
weak evidence for different weak evidence hides the problem instead of fixing it.

**A successful fallback is not a satisfied goal.** Ask for the weather in Kano,
watch the fallback return a weather *news article*, and watch the critic catch
that the evidence never mentions Kano. This is the failure mode that produces
fluent, well-cited, wrong answers.

**Reflection is a checklist, not a mood.** Did every step run? Did anything
fail? Was any retrieval weak? Does the evidence mention the subject matter of
the goal? Did a dependent step run without its inputs? Each failed check becomes
a *gap*, phrased in the planner's input language — that is the entire replanning
mechanism.

**Gaps must be actionable, and loops must be bounded.** Two independent brakes:
a hard iteration budget, and a progress check that stops when a replan produces
no gap the agent has not already failed at. Spinning looks exactly like working
until you read the trace.

**The trace is the product.** Control flow changes every run, so a stack trace
and a print statement are not enough. Every line of console output here comes
from the tracer, and the same run dumps to JSON for an eval harness.

**Specialisation by capability, not instruction.** The researcher cannot do
arithmetic because `calculator` is not in its registry. Constraining the toolbox
is far more reliable than asking a component nicely.

**More agents is not better.** Lesson 8 says so out loud, and the corpus says so
too. You pay in calls, latency and coordination failure modes. The payoff is
role clarity and an independent review step — not headcount.

## Things worth breaking

The demo is small enough to modify in one sitting. Suggested exercises, roughly
in order of difficulty:

1. Add a `currency` tool with its own failure kinds, and wire it into the router
   and the fallback map.
2. Drop your own `.txt` files into `data/corpus/` and watch the retrieval scores
   move. Find a query the grader marks weak that you think should be strong.
3. Set `WeatherTool(offline=True)` and delete `web_search` from the registry.
   The agent should degrade honestly rather than inventing a temperature.
4. Break the argument binding in `Agent._resolve_dependencies` and see which
   test catches it. (`test_dependent_step_gets_bound_arguments`.)
5. Raise `RagSearchTool.threshold` to 0.9. Every retrieval becomes weak — watch
   what the critic and the fallback chain do about it.
6. Add a sixth check to the `Critic`: flag when two observations contradict each
   other on the same fact.
7. Replace `KeywordRouter` with a real LLM call. The interface is
   `route(query) -> (tool_name, reason)`. Nothing else in the project has to
   change, which is the point.

## Testing an agent

The tests assert the *trajectory*, not just the answer, because two runs can
reach the same output through very different paths and only one of them is
reliable:

```python
def test_transient_failure_is_retried_and_succeeds(self):
    registry = build_default_registry(weather_fail_first=1)
    obs = Executor(registry, QUIET()).run_step(...)
    self.assertTrue(obs.ok)
    self.assertEqual(obs.attempts, 2)       # proof the retry actually happened
    self.assertIsNone(obs.recovered_with)   # and that no fallback was needed
```

Failures are injected on a fixed schedule rather than at random, so a recovery
test passes or fails for a real reason instead of flapping. Two tests also
assert the *documented bugs* in `NaivePlanner`, so nobody quietly "fixes" the
thing Lesson 1 exists to demonstrate.

## Where the illusion ends

Being honest about what this is not: the router is keyword rules, the retrieval
is TF-IDF, and the answers are assembled from templates. A real agent replaces
all three with a model. What does not change is the shape — the loop, the tool
contracts, the error taxonomy, the grading step, the critic, the trace, the
bounded replanning. Swapping in an LLM changes the quality of each decision, not
the architecture around it. That is why the architecture is what this project
teaches.
