"""
Calculator: safe arithmetic without `eval`.

Teaching point
--------------
This is the smallest example of a *guardrail*. `eval("2+2")` works, and
`eval("__import__('os').system('rm -rf /')")` also works. When a planner
(or an LLM) is choosing the argument string, you must not hand that string
to an interpreter. So we parse it into an AST and walk only the node types
we have explicitly allowed.

The same principle scales up: allow-list the operations, never deny-list
the attacks.
"""

from __future__ import annotations

import ast
import operator

from . import Tool, ToolResult
from ..errors import ToolError

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Word forms a planner might have extracted straight from the user's phrasing.
_WORD_OPS = {
    " plus ": " + ",
    " minus ": " - ",
    " times ": " * ",
    " multiplied by ": " * ",
    " divided by ": " / ",
    " over ": " / ",
    "^": "**",
    "x": "*",
}


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate a single arithmetic expression over numbers. Supports "
        "+ - * / // % and **. Does NOT look anything up; the caller must "
        "supply concrete numbers."
    )
    args_schema = {"expression": "An arithmetic expression, e.g. '(31 - 28) * 1.8'"}
    examples = ("calculate 12 * 7", "what is 100 / 4 minus 3")

    MAX_POWER = 64  # stop 9**9**9 from hanging the demo

    def run(self, expression: str = "", **_) -> ToolResult:
        raw = (expression or "").strip()
        if not raw:
            raise ToolError(
                "no expression supplied",
                kind="invalid_input",
                hint="extract the numbers from the goal before calling calculator",
                tool=self.name,
            )
        cleaned = self._normalise(raw)
        try:
            tree = ast.parse(cleaned, mode="eval")
        except SyntaxError as exc:
            raise ToolError(
                f"cannot parse expression {cleaned!r}",
                kind="invalid_input",
                hint="strip any words, leave only numbers and operators",
                tool=self.name,
            ) from exc

        value = self._eval(tree.body)
        pretty = round(value, 6) if isinstance(value, float) else value
        return ToolResult(
            summary=f"{cleaned} = {pretty}",
            data={"expression": cleaned, "value": pretty},
            citations=["[calculator]"],
            meta={"raw_input": raw},
        )

    # -- internals --------------------------------------------------------

    def _normalise(self, text: str) -> str:
        out = f" {text.lower()} "
        for word, symbol in _WORD_OPS.items():
            out = out.replace(word, symbol)
        # Drop anything that is not part of an arithmetic expression.
        allowed = set("0123456789.+-*/%() ")
        out = "".join(ch for ch in out if ch in allowed)
        return " ".join(out.split())

    def _eval(self, node: ast.AST):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ToolError(
                f"unsupported literal: {node.value!r}", kind="invalid_input", tool=self.name
            )
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > self.MAX_POWER:
                raise ToolError(
                    "exponent too large", kind="invalid_input", tool=self.name
                )
            try:
                return _BINOPS[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise ToolError(
                    "division by zero",
                    kind="invalid_input",
                    hint="check the denominator before dividing",
                    tool=self.name,
                ) from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](self._eval(node.operand))
        raise ToolError(
            f"operation not allowed: {type(node).__name__}",
            kind="invalid_input",
            hint="only + - * / // % ** on plain numbers are permitted",
            tool=self.name,
        )
