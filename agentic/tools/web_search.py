"""
Web search over a simulated, offline "internet" (data/web/*.txt).

Teaching point
--------------
Note how similar this is to the RAG tool -- same index, same ranking. The
difference that matters to the *planner* is the description: this source is
recent and external, the RAG source is curated and internal. Routing between
two tools that share an implementation but differ in freshness and authority
is an extremely common real-world design.
"""

from __future__ import annotations

from pathlib import Path

from . import Tool, ToolResult
from ..errors import ToolError
from ..retrieval import VectorIndex


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search a snapshot of recent news articles, each stamped with a "
        "publication date. Use for anything time-sensitive: latest, today, "
        "recent, current, this week. Not authoritative for definitions."
    )
    args_schema = {"query": "Search keywords drawn from the goal"}
    examples = ("latest news on agent frameworks", "what happened today")

    def __init__(self, web_dir: str | Path, offline: bool = False) -> None:
        self.web_dir = Path(web_dir)
        self.offline = offline
        self._index: VectorIndex | None = None

    @property
    def index(self) -> VectorIndex:
        if self._index is None:  # lazy: don't read disk until first use
            self._index = VectorIndex.from_directory(self.web_dir)
        return self._index

    def run(self, query: str = "", k: int = 3, **_) -> ToolResult:
        if self.offline:
            raise ToolError(
                "search backend unreachable", kind="unavailable", tool=self.name
            )
        if not query.strip():
            raise ToolError("empty query", kind="invalid_input", tool=self.name)

        hits = self.index.search(query, k=k)
        if not hits:
            raise ToolError(
                f"no results for {query!r}",
                kind="not_found",
                hint="try broader keywords",
                tool=self.name,
            )

        lines, cites = [], []
        for h in hits:
            date = h.passage.meta.get("date", "undated")
            title = h.passage.meta.get("title", h.passage.doc_id)
            lines.append(f"({date}) {title} :: {h.passage.preview(140)}")
            cites.append(h.passage.citation)

        return ToolResult(
            summary=f"{len(hits)} article(s); top: {hits[0].passage.meta.get('title', '?')}",
            data={
                "results": lines,
                "top_score": hits[0].score,
                "dates": [h.passage.meta.get("date") for h in hits],
            },
            citations=cites,
            grade="strong" if hits[0].score >= 0.12 else "weak",
            meta={"scores": [h.score for h in hits]},
        )
