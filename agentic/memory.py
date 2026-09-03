"""
Three kinds of memory, kept deliberately separate.

    Scratchpad      working memory -- this run only, thrown away at the end
    EpisodicMemory  past runs -- persisted to JSON, survives restarts
    VectorIndex     semantic memory -- the knowledge base (see retrieval.py)

Teaching point
--------------
Most "my agent is behaving inconsistently" bugs are memory bugs. Keeping the
three stores in separate objects with separate lifetimes makes it obvious
which one is responsible when behaviour changes between runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .retrieval import tokenize


@dataclass
class Scratchpad:
    """Working memory for a single run."""

    goal: str = ""
    facts: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    failed_tools: set[str] = field(default_factory=set)

    def remember_fact(self, key: str, value: Any, source: str = "") -> None:
        self.facts.append({"key": key, "value": value, "source": source})

    def lookup(self, key: str) -> Any | None:
        for fact in reversed(self.facts):
            if fact["key"] == key:
                return fact["value"]
        return None

    def numeric_facts(self) -> list[dict]:
        return [f for f in self.facts if isinstance(f["value"], (int, float))]

    def as_evidence(self) -> str:
        return "\n".join(f"- {f['key']}: {f['value']} {f['source']}" for f in self.facts)


@dataclass
class Episode:
    goal: str
    plan: list[str]
    tools_used: list[str]
    success: bool
    gaps: list[str]
    answer: str
    duration_ms: float
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return asdict(self)


class EpisodicMemory:
    """Past runs, on disk as JSON. No database, no dependencies."""

    def __init__(self, path: str | Path | None = None, autoload: bool = True) -> None:
        self.path = Path(path) if path else None
        self.episodes: list[Episode] = []
        if autoload and self.path and self.path.exists():
            self.load()

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if self.path:
            self.save()

    def recall_similar(self, goal: str, threshold: float = 0.55) -> Episode | None:
        """Jaccard overlap on content words -- crude, transparent, adequate.

        The point is not the similarity metric. The point is that an agent
        which recognises a goal it has already solved can skip rediscovery.
        """
        target = set(tokenize(goal))
        if not target:
            return None
        best, best_score = None, 0.0
        for ep in self.episodes:
            other = set(tokenize(ep.goal))
            if not other:
                continue
            score = len(target & other) / len(target | other)
            if score > best_score:
                best, best_score = ep, score
        if best is not None and best_score >= threshold and best.success:
            return best
        return None

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [e.as_dict() for e in self.episodes][-200:]  # bounded: forgetting is a feature
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self.episodes = [Episode(**item) for item in raw]

    def clear(self) -> None:
        self.episodes = []
        if self.path and self.path.exists():
            self.path.unlink()

    def stats(self) -> dict:
        total = len(self.episodes)
        wins = sum(1 for e in self.episodes if e.success)
        return {
            "episodes": total,
            "success_rate": round(wins / total, 2) if total else 0.0,
        }
