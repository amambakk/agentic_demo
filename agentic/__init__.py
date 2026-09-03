"""
agentic -- an offline, standard-library-only teaching implementation of an
agentic AI system.

No LLM. No network. No API key. No third-party package. Every decision that
a language model would normally make is made instead by a rule you can read,
so the *architecture* is visible rather than hidden behind a prompt.

Quick start:

    from agentic import build_agent
    agent = build_agent()
    print(agent.solve("what is corrective rag").answer)
"""

from .agent import Agent, AgentResult, Composer
from .errors import ToolError
from .executor import Executor, Observation
from .memory import Episode, EpisodicMemory, Scratchpad
from .multi_agent import Blackboard, Entry, Specialist, Team
from .planner import (
    DecompositionPlanner,
    KeywordRouter,
    NaivePlanner,
    Plan,
    Step,
)
from .reflection import Critic, Reflection
from .retrieval import Passage, VectorIndex, tokenize
from .tools import ToolRegistry, ToolResult, build_default_registry
from .trace import Tracer, rule

__version__ = "1.0.0"

__all__ = [
    "Agent", "AgentResult", "Composer", "Critic", "Reflection",
    "DecompositionPlanner", "KeywordRouter", "NaivePlanner", "Plan", "Step",
    "Executor", "Observation", "ToolError",
    "EpisodicMemory", "Episode", "Scratchpad",
    "Team", "Blackboard", "Entry", "Specialist",
    "VectorIndex", "Passage", "tokenize",
    "ToolRegistry", "ToolResult", "build_default_registry",
    "Tracer", "rule", "build_agent", "build_team",
]


def build_agent(trace: bool = True, weather_fail_first: int = 0, memory_path=None) -> Agent:
    """Convenience constructor with the default toolbox."""
    registry = build_default_registry(weather_fail_first=weather_fail_first)
    return Agent(
        registry=registry,
        tracer=Tracer(enabled=trace),
        episodic=EpisodicMemory(memory_path),
    )


def build_team(trace: bool = True) -> Team:
    return Team(build_default_registry(), Tracer(enabled=trace))
