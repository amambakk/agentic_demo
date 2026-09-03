"""
Weather: a canned dataset that fails on purpose.

Teaching point
--------------
Recovery code you have never seen execute is recovery code that does not
work. This tool injects failures on a *deterministic schedule* rather than
at random, so:

    fail_first_n_calls=1  -> attempt 1 raises `transient`, attempt 2 succeeds
                             (demonstrates retry)
    an unknown city       -> raises `not_found`, which must NOT be retried
                             (demonstrates fallback to another tool)
    offline=True          -> raises `unavailable` every time
                             (demonstrates routing around a dead dependency)

Deterministic injection is also what makes the recovery test in tests/
meaningful instead of flaky.
"""

from __future__ import annotations

from . import Tool, ToolResult
from ..errors import ToolError

# Fixed snapshot. No network, no API key, no clock dependency.
WEATHER_DB: dict[str, dict] = {
    "lagos":    {"temp_c": 31, "condition": "humid, scattered thunderstorms", "humidity": 84},
    "accra":    {"temp_c": 28, "condition": "warm, partly cloudy", "humidity": 78},
    "nairobi":  {"temp_c": 22, "condition": "mild, overcast", "humidity": 61},
    "london":   {"temp_c": 17, "condition": "drizzle", "humidity": 72},
    "berlin":   {"temp_c": 19, "condition": "clear", "humidity": 55},
    "tokyo":    {"temp_c": 27, "condition": "hot, high cloud", "humidity": 70},
    "new york": {"temp_c": 24, "condition": "sunny", "humidity": 48},
}


class WeatherTool(Tool):
    name = "weather"
    description = (
        "Look up current temperature and conditions for one named city from an "
        "offline snapshot. Covers major cities only; returns not_found for "
        "anything else. Cannot forecast and cannot compare two cities."
    )
    args_schema = {"city": "A single city name, e.g. 'Lagos'"}
    examples = ("what is the weather in Accra", "temperature in Tokyo")

    def __init__(self, fail_first_n_calls: int = 0, offline: bool = False) -> None:
        self.fail_first_n_calls = fail_first_n_calls
        self.offline = offline
        self.calls = 0

    def run(self, city: str = "", **_) -> ToolResult:
        self.calls += 1

        if self.offline:
            raise ToolError(
                "weather service is unreachable",
                kind="unavailable",
                hint="route around this tool for the rest of the run",
                tool=self.name,
            )

        # Injected transient failure on the first N calls of the process.
        if self.calls <= self.fail_first_n_calls:
            raise ToolError(
                f"upstream timeout (attempt {self.calls})",
                kind="transient",
                hint="retry with backoff",
                tool=self.name,
            )

        key = (city or "").strip().lower()
        if not key:
            raise ToolError(
                "no city supplied",
                kind="invalid_input",
                hint="extract a city name from the goal first",
                tool=self.name,
            )
        if key not in WEATHER_DB:
            raise ToolError(
                f"no coverage for {city!r}",
                kind="not_found",
                hint=f"covered cities: {', '.join(sorted(WEATHER_DB))}",
                tool=self.name,
            )

        row = WEATHER_DB[key]
        pretty = city.strip().title()
        return ToolResult(
            summary=f"{pretty}: {row['temp_c']}C, {row['condition']} (humidity {row['humidity']}%)",
            data={"city": pretty, **row},
            citations=[f"[weather:{key}]"],
            meta={"attempt": self.calls},
        )

    def reset(self) -> None:
        self.calls = 0
