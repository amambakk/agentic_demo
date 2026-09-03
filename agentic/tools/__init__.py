"""
The tool contract.

Teaching point
--------------
A tool is four things, and only one of them is code:

    1. a name          the planner routes to it
    2. a description   the planner decides *when* to use it from this text
    3. an arg schema   the planner has to fill these in correctly
    4. a failure mode  structured, classifiable, recoverable

In an LLM-driven system, items 1-3 are literally what gets serialised into
the function-calling schema the model sees. `Tool.spec()` below produces
that shape, so you can see exactly what the model would be reading. Vague
descriptions produce bad routing; that is a prompt bug, not a model bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import ToolError


@dataclass
class ToolResult:
    """What a successful tool call returns."""

    summary: str                       # one line, for the trace and the answer
    data: Any = None                   # structured payload for downstream steps
    citations: list[str] = field(default_factory=list)
    grade: str = "strong"              # strong | weak  -- see corrective RAG
    meta: dict = field(default_factory=dict)


class Tool:
    name: str = "tool"
    description: str = ""
    args_schema: dict[str, str] = {}
    examples: tuple[str, ...] = ()

    def run(self, **kwargs) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError

    def spec(self) -> dict:
        """The JSON-ish shape an LLM would be shown for function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    k: {"type": "string", "description": v}
                    for k, v in self.args_schema.items()
                },
                "required": list(self.args_schema),
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Tool {self.name}>"


class ToolRegistry:
    """The agent's toolbox. Also the thing a planner is allowed to route to."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(
                f"no such tool: {name!r}",
                kind="unavailable",
                hint=f"available: {', '.join(sorted(self._tools))}",
                tool=name,
            )
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def subset(self, names: list[str]) -> "ToolRegistry":
        """A narrowed toolbox -- used to give sub-agents a smaller surface."""
        return ToolRegistry([self._tools[n] for n in names if n in self._tools])

    def catalog(self) -> str:
        lines = []
        for name in self.names():
            tool = self._tools[name]
            args = ", ".join(tool.args_schema) or "-"
            lines.append(f"  {name:<14} args({args})\n      {tool.description}")
        return "\n".join(lines)

    def specs(self) -> list[dict]:
        return [self._tools[n].spec() for n in self.names()]

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


from .calculator import CalculatorTool          # noqa: E402
from .weather import WeatherTool                # noqa: E402
from .web_search import WebSearchTool           # noqa: E402
from .rag_search import RagSearchTool           # noqa: E402
from .notepad import NotepadTool                # noqa: E402

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "ToolError",
    "CalculatorTool",
    "WeatherTool",
    "WebSearchTool",
    "RagSearchTool",
    "NotepadTool",
    "build_default_registry",
]


def build_default_registry(
    corpus_dir: str | None = None,
    web_dir: str | None = None,
    weather_fail_first: int = 0,
) -> ToolRegistry:
    """Assemble the standard toolbox used by the lessons."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "data"
    corpus_dir = corpus_dir or str(root / "corpus")
    web_dir = web_dir or str(root / "web")
    return ToolRegistry(
        [
            CalculatorTool(),
            WeatherTool(fail_first_n_calls=weather_fail_first),
            WebSearchTool(web_dir),
            RagSearchTool(corpus_dir),
            NotepadTool(),
        ]
    )
