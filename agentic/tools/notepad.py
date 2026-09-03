"""
Notepad: a tool whose side effect is a memory write.

Teaching point
--------------
Not every tool reads the world; some change it. A write tool is where the
"agent" part gets genuinely risky, because a bad plan now has consequences
that outlive the run. Two habits worth forming early, both visible here:

    * write tools log their side effect into the trace explicitly
    * write tools are separate from read tools, so a read-only sub-agent
      can be handed a registry that simply does not contain them
"""

from __future__ import annotations

from . import Tool, ToolResult
from ..errors import ToolError


class NotepadTool(Tool):
    name = "notepad"
    description = (
        "Store or recall a short named note. action='write' needs key and "
        "value; action='read' needs key. This is the only tool with a side "
        "effect."
    )
    args_schema = {
        "action": "'write' or 'read'",
        "key": "Short identifier for the note",
        "value": "Text to store (write only)",
    }

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def run(self, action: str = "read", key: str = "", value: str = "", **_) -> ToolResult:
        key = key.strip()
        if not key:
            raise ToolError("no key supplied", kind="invalid_input", tool=self.name)

        if action == "write":
            self.store[key] = value
            return ToolResult(
                summary=f"note {key!r} saved ({len(value)} chars)",
                data={"key": key, "written": True},
                citations=[f"[note:{key}]"],
                meta={"side_effect": "memory_write"},
            )

        if action == "read":
            if key not in self.store:
                raise ToolError(
                    f"no note called {key!r}",
                    kind="not_found",
                    hint=f"known notes: {', '.join(sorted(self.store)) or 'none'}",
                    tool=self.name,
                )
            return ToolResult(
                summary=f"note {key!r}: {self.store[key]}",
                data={"key": key, "value": self.store[key]},
                citations=[f"[note:{key}]"],
            )

        raise ToolError(
            f"unknown action {action!r}",
            kind="invalid_input",
            hint="action must be 'read' or 'write'",
            tool=self.name,
        )
