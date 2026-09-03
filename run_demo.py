#!/usr/bin/env python3
"""
Entry point for the offline agentic AI teaching demo.

    python run_demo.py                       guided tour (start here)
    python run_demo.py --lessons             list the nine lessons
    python run_demo.py --lesson 6            run one lesson
    python run_demo.py --all                 run every lesson end to end
    python run_demo.py --ask "..."           single agent, traced
    python run_demo.py --team "..."          multi-agent team, traced
    python run_demo.py --repl                interactive loop
    python run_demo.py --tools               print the tool catalog
    python run_demo.py --trace-json out.json save the trace of --ask/--team
    python run_demo.py --quiet               answer only, no trace
    python run_demo.py --fail-weather 1      inject N transient weather failures
    python run_demo.py --test                run the unit tests

No network. No API key. No third-party packages. Python 3.10+.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentic import lessons  # noqa: E402
from agentic.agent import Agent  # noqa: E402
from agentic.memory import EpisodicMemory  # noqa: E402
from agentic.multi_agent import Team  # noqa: E402
from agentic.tools import build_default_registry  # noqa: E402
from agentic.trace import Tracer, rule  # noqa: E402

MEMORY_PATH = Path("./.agent_memory/episodes.json")

SAMPLES = [
    "what is corrective rag",
    "compare the weather in Lagos and Accra and give the temperature difference",
    "what is the latest news on agent frameworks",
    "calculate 145 * 3 minus 12",
    "what is the weather in Kano",
    "explain how multi agent systems coordinate, and what is the weather in Tokyo",
]


def banner() -> None:
    rule("OFFLINE AGENTIC AI DEMO", char="=")
    print("  Goal -> Plan -> Act -> Observe -> Reflect -> Respond")
    print("  No LLM, no network, no API key, no dependencies.")
    print("  Every decision an LLM would make is a rule you can read.\n")


def guided_tour() -> None:
    banner()
    lessons.list_lessons()
    print()
    rule("A 30-SECOND TASTE", char="-")
    agent = Agent(build_default_registry(weather_fail_first=1), Tracer())
    result = agent.solve(
        "compare the weather in Lagos and Accra and give the temperature difference"
    )
    print("\n--- ANSWER " + "-" * 66)
    print(result.answer)
    print("\nRead that trace top to bottom, then run:  python run_demo.py --lesson 1")


def ask(question: str, args) -> None:
    tracer = Tracer(enabled=not args.quiet)
    agent = Agent(
        build_default_registry(weather_fail_first=args.fail_weather),
        tracer,
        episodic=EpisodicMemory(MEMORY_PATH if args.remember else None),
    )
    result = agent.solve(question)
    print("\n--- ANSWER " + "-" * 66)
    print(result.answer)
    if not args.quiet:
        print("\nMetrics:", tracer.summary())
    _maybe_save(tracer, args)


def team(question: str, args) -> None:
    tracer = Tracer(enabled=not args.quiet)
    out = Team(build_default_registry(), tracer).solve(question)
    print("\n--- ANSWER " + "-" * 66)
    print(out["answer"])
    if not args.quiet:
        print("\nMetrics:", tracer.summary())
    _maybe_save(tracer, args)


def _maybe_save(tracer: Tracer, args) -> None:
    if args.trace_json:
        tracer.save(args.trace_json)
        print(f"\nTrace written to {args.trace_json}")


def repl(args) -> None:
    banner()
    print("Interactive mode. Try one of these, or type your own:")
    for s in SAMPLES:
        print(f"   - {s}")
    print("\nPrefix with 'team:' to use the multi-agent team. Ctrl-D or 'quit' to exit.\n")
    while True:
        try:
            line = input("goal> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            return
        print()
        if line.lower().startswith("team:"):
            team(line[5:].strip(), args)
        else:
            ask(line, args)
        print()


def run_tests() -> int:
    import unittest

    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).parent / "tests"), top_level_dir=str(Path(__file__).parent))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="Offline agentic AI teaching demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--lessons", action="store_true", help="list the lessons")
    p.add_argument("--lesson", type=int, metavar="N", help="run lesson N")
    p.add_argument("--all", action="store_true", help="run every lesson")
    p.add_argument("--ask", metavar="GOAL", help="run the single agent on a goal")
    p.add_argument("--team", metavar="GOAL", help="run the multi-agent team on a goal")
    p.add_argument("--repl", action="store_true", help="interactive prompt")
    p.add_argument("--tools", action="store_true", help="print the tool catalog")
    p.add_argument("--trace-json", metavar="PATH", help="write the trace as JSON")
    p.add_argument("--quiet", action="store_true", help="suppress the trace")
    p.add_argument("--remember", action="store_true", help="persist episodes to disk")
    p.add_argument("--fail-weather", type=int, default=0, metavar="N",
                   help="inject N transient weather failures")
    p.add_argument("--test", action="store_true", help="run the unit tests")
    args = p.parse_args()

    if args.test:
        return run_tests()
    if args.tools:
        print(build_default_registry().catalog())
        return 0
    if args.lessons:
        lessons.list_lessons()
        return 0
    if args.lesson:
        lessons.run_lesson(args.lesson)
        return 0
    if args.all:
        lessons.run_all()
        return 0
    if args.ask:
        ask(args.ask, args)
        return 0
    if args.team:
        team(args.team, args)
        return 0
    if args.repl:
        repl(args)
        return 0
    guided_tour()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
