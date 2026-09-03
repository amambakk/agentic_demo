"""
Tests. Note what is being asserted: not just the final answer, but the
TRAJECTORY -- did it retry, did it fall back, did it refuse to retry a
non-retryable error. That is agent testing.

    python run_demo.py --test
    python -m unittest discover tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.agent import Agent
from agentic.errors import ToolError
from agentic.executor import Executor
from agentic.memory import Episode, EpisodicMemory
from agentic.planner import DecompositionPlanner, KeywordRouter, NaivePlanner, Step
from agentic.reflection import Critic
from agentic.retrieval import VectorIndex, tokenize
from agentic.tools import build_default_registry
from agentic.trace import Tracer

QUIET = lambda: Tracer(enabled=False)  # noqa: E731


class TestRouting(unittest.TestCase):
    def test_naive_router_is_case_sensitive(self):
        self.assertEqual(NaivePlanner("calculate 2+2").which_tool_to_choose(), "calculator")
        # The documented bug, asserted so it cannot be "fixed" by accident.
        self.assertEqual(NaivePlanner("Calculate 2+2").which_tool_to_choose(), "rag_search")

    def test_naive_router_matches_substrings(self):
        self.assertEqual(
            NaivePlanner("is there a newsletter").which_tool_to_choose(), "web_search"
        )

    def test_fixed_router_handles_both(self):
        r = KeywordRouter()
        self.assertEqual(r.route("Calculate 2+2")[0], "calculator")
        self.assertEqual(r.route("is there a newsletter")[0], "rag_search")

    def test_router_defaults_to_knowledge_base(self):
        tool, reason = KeywordRouter().route("explain the agent loop")
        self.assertEqual(tool, "rag_search")
        self.assertIn("default", reason)


class TestPlanner(unittest.TestCase):
    def test_comparison_produces_three_ordered_steps(self):
        plan = DecompositionPlanner().plan(
            "compare the weather in Lagos and Accra and give the difference"
        )
        self.assertEqual(plan.tools(), ["weather", "weather", "calculator"])
        self.assertEqual(plan.steps[2].depends_on, [1, 2])

    def test_city_extraction(self):
        plan = DecompositionPlanner().plan("what is the weather in Tokyo")
        self.assertEqual(plan.steps[0].args["city"], "Tokyo")

    def test_gaps_drive_a_new_plan(self):
        plan = DecompositionPlanner().plan("anything", gaps=["latest news on tracing"])
        self.assertEqual(plan.steps[0].tool, "web_search")
        self.assertIn("gap", plan.steps[0].reason)


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = VectorIndex.from_directory(
            Path(__file__).resolve().parents[1] / "data" / "corpus"
        )

    def test_tokenizer_drops_stopwords(self):
        self.assertNotIn("the", tokenize("the agent loop"))

    def test_index_built(self):
        self.assertGreater(self.index.stats()["passages"], 10)

    def test_relevant_query_ranks_right_document(self):
        hits = self.index.search("corrective retrieval grading threshold", k=1)
        self.assertEqual(hits[0].passage.doc_id, "retrieval.txt")

    def test_offtopic_query_scores_low(self):
        on = self.index.search("multi agent blackboard coordination", k=1)[0].score
        off = self.index.search("capital city of Mongolia", k=1)
        off_score = off[0].score if off else 0.0
        self.assertGreater(on, off_score)


class TestTools(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_registry()

    def test_calculator_evaluates(self):
        self.assertEqual(self.registry.get("calculator").run(expression="12 * 7").data["value"], 84)

    def test_calculator_handles_word_operators(self):
        self.assertEqual(
            self.registry.get("calculator").run(expression="10 plus 5").data["value"], 15
        )

    def test_calculator_blocks_code_execution(self):
        with self.assertRaises(ToolError):
            self.registry.get("calculator").run(expression="__import__('os').system('ls')")

    def test_divide_by_zero_is_invalid_input_not_transient(self):
        with self.assertRaises(ToolError) as ctx:
            self.registry.get("calculator").run(expression="1/0")
        self.assertEqual(ctx.exception.kind, "invalid_input")
        self.assertFalse(ctx.exception.retryable)

    def test_weather_unknown_city_is_not_found(self):
        with self.assertRaises(ToolError) as ctx:
            self.registry.get("weather").run(city="Atlantis")
        self.assertEqual(ctx.exception.kind, "not_found")

    def test_rag_grades_offtopic_query_weak(self):
        rag = self.registry.get("rag_search")
        self.assertEqual(rag.run(query="agent reflection critic").grade, "strong")
        self.assertEqual(rag.run(query="tell me about jollof rice recipes").grade, "weak")

    def test_notepad_round_trip(self):
        pad = self.registry.get("notepad")
        pad.run(action="write", key="k", value="v")
        self.assertEqual(pad.run(action="read", key="k").data["value"], "v")

    def test_registry_subset_narrows_capability(self):
        subset = self.registry.subset(["rag_search"])
        self.assertTrue(subset.has("rag_search"))
        self.assertFalse(subset.has("calculator"))


class TestRecovery(unittest.TestCase):
    def test_transient_failure_is_retried_and_succeeds(self):
        registry = build_default_registry(weather_fail_first=1)
        ex = Executor(registry, QUIET())
        obs = ex.run_step(Step(intent="weather", tool="weather", args={"city": "Lagos"}, id=1))
        self.assertTrue(obs.ok)
        self.assertEqual(obs.attempts, 2)          # proof the retry happened
        self.assertIsNone(obs.recovered_with)      # no fallback needed

    def test_not_found_falls_back_instead_of_retrying(self):
        ex = Executor(build_default_registry(), QUIET())
        obs = ex.run_step(Step(intent="weather in Kano", tool="weather",
                               args={"city": "Kano"}, id=1))
        self.assertEqual(obs.attempts, 1)          # NOT retried
        if obs.ok:
            self.assertEqual(obs.recovered_with, "web_search")

    def test_unavailable_tool_is_marked_dead(self):
        registry = build_default_registry()
        registry.get("weather").offline = True
        ex = Executor(registry, QUIET())
        ex.run_step(Step(intent="w", tool="weather", args={"city": "Lagos"}, id=1))
        self.assertIn("weather", ex.dead_tools)

    def test_fallback_translates_arguments(self):
        step = Step(intent="weather in Kano", tool="weather", args={"city": "Kano"}, id=1)
        self.assertEqual(
            Executor._translate_args(step, "web_search"), {"query": "Kano weather"}
        )


class TestReflection(unittest.TestCase):
    def test_failed_step_becomes_a_gap(self):
        registry = build_default_registry()
        registry.get("weather").offline = True
        registry._tools.pop("web_search")  # remove the fallback too
        agent = Agent(registry, QUIET())
        result = agent.solve("what is the weather in Lagos")
        self.assertFalse(result.reflection.complete)
        self.assertTrue(result.reflection.issues)

    def test_clean_run_is_complete(self):
        agent = Agent(build_default_registry(), QUIET())
        result = agent.solve("what is the weather in Lagos")
        self.assertTrue(result.reflection.complete)
        self.assertEqual(result.iterations, 1)

    def test_critic_bounds_the_gap_list(self):
        c = Critic()
        self.assertLessEqual(len(c.review("x", type("P", (), {"steps": []})(), []).gaps), 3)


class TestAgentLoop(unittest.TestCase):
    def test_dependent_step_gets_bound_arguments(self):
        agent = Agent(build_default_registry(), QUIET())
        result = agent.solve("compare the weather in Lagos and Accra and give the difference")
        calc = [o for o in result.observations if o.tool == "calculator"]
        self.assertTrue(calc and calc[0].ok)
        self.assertEqual(calc[0].data["value"], 3)     # 31C - 28C

    def test_iterations_are_bounded(self):
        agent = Agent(build_default_registry(), QUIET(), max_iterations=2)
        result = agent.solve("tell me about quantum tunnelling in llamas")
        self.assertLessEqual(result.iterations, 2)

    def test_answer_carries_citations(self):
        agent = Agent(build_default_registry(), QUIET())
        result = agent.solve("what is corrective rag")
        self.assertTrue(result.citations)
        self.assertIn("Sources:", result.answer)

    def test_trace_records_every_stage(self):
        tracer = Tracer(enabled=False)
        Agent(build_default_registry(), tracer).solve("what is the agent loop")
        stages = {e.stage for e in tracer.events}
        for expected in {"GOAL", "PLAN", "ACT", "OBSERVE", "REFLECT", "RESPOND", "MEMORY"}:
            self.assertIn(expected, stages)

    def test_trace_serialises_to_json(self):
        tracer = Tracer(enabled=False)
        Agent(build_default_registry(), tracer).solve("what is memory")
        self.assertIn('"stage"', tracer.to_json())


class TestMemory(unittest.TestCase):
    def test_similar_goal_is_recalled(self):
        mem = EpisodicMemory()
        mem.add(Episode(goal="explain the agent loop", plan=[], tools_used=["rag_search"],
                        success=True, gaps=[], answer="...", duration_ms=1.0))
        self.assertIsNotNone(mem.recall_similar("explain the agent loop"))
        self.assertIsNone(mem.recall_similar("weather in Tokyo tomorrow"))

    def test_failed_episodes_are_not_reused(self):
        mem = EpisodicMemory()
        mem.add(Episode(goal="explain the agent loop", plan=[], tools_used=[],
                        success=False, gaps=["x"], answer="", duration_ms=1.0))
        self.assertIsNone(mem.recall_similar("explain the agent loop"))

    def test_agent_records_an_episode(self):
        mem = EpisodicMemory()
        Agent(build_default_registry(), QUIET(), episodic=mem).solve("what is planning")
        self.assertEqual(len(mem.episodes), 1)


class TestMultiAgent(unittest.TestCase):
    def test_specialists_have_narrowed_toolboxes(self):
        from agentic.multi_agent import Team

        team = Team(build_default_registry(), QUIET())
        self.assertFalse(team.researcher.registry.has("calculator"))
        self.assertFalse(team.analyst.registry.has("rag_search"))

    def test_team_produces_a_cited_answer(self):
        from agentic.multi_agent import Team

        out = Team(build_default_registry(), QUIET()).solve(
            "explain multi agent coordination, and the weather in Lagos"
        )
        self.assertIn("Sources:", out["answer"])
        authors = {e.author for e in out["blackboard"].entries}
        self.assertIn("critic", authors)
        self.assertIn("writer", authors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
