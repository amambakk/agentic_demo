"""
Structured failures.

Teaching point
--------------
An agent can only recover from a failure it can *classify*. A tool that
returns the string "sorry, something went wrong" is indistinguishable from
a tool that returned an answer. So every tool in this project raises a
`ToolError` carrying a `kind`, and the executor picks a recovery strategy
from that kind:

    transient      -> retry with backoff (same arguments)
    invalid_input  -> do NOT retry; repair the arguments first
    not_found      -> do NOT retry; fall back to a broader source
    unavailable    -> do NOT retry; route around the dead dependency
"""

from __future__ import annotations

RETRYABLE_KINDS = {"transient"}


class ToolError(Exception):
    """A failure raised by a tool, with enough structure to recover from."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "unknown",
        hint: str | None = None,
        tool: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.hint = hint
        self.tool = tool

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS

    def as_dict(self) -> dict:
        return {
            "tool": self.tool,
            "kind": self.kind,
            "message": self.message,
            "hint": self.hint,
            "retryable": self.retryable,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.kind}] {self.message}"


class BudgetExceeded(Exception):
    """Raised when the agent runs out of steps, retries, or wall time."""
