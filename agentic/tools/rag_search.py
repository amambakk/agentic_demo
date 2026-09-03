"""
RAG search over the local knowledge base (data/corpus/*.txt), with grading.

Teaching point -- Corrective RAG
--------------------------------
Naive RAG retrieves the top k passages and answers from them *no matter how
bad they are*. Cosine similarity always returns something; "something" is
not the same as "relevant". That is the mechanism behind a large share of
confident, well-cited, completely wrong answers.

So this tool grades itself. If the best score is below `threshold` it still
returns the passages, but marks `grade="weak"`. The agent then has a real
decision to make: rewrite the query, fall back to web_search, or admit the
knowledge base does not cover the question. That decision lives in the
agent (see agent.py), not here -- the tool reports, the agent decides.
"""

from __future__ import annotations

from pathlib import Path

from . import Tool, ToolResult
from ..errors import ToolError
from ..retrieval import VectorIndex


class RagSearchTool(Tool):
    name = "rag_search"
    description = (
        "Retrieve passages from the curated internal knowledge base about "
        "agentic AI: the agent loop, planning, tool use, memory, retrieval, "
        "reflection, failure recovery, multi-agent design and evaluation. "
        "Authoritative but static; knows nothing about current events."
    )
    args_schema = {"query": "The sub-question to look up, in natural language"}
    examples = ("what is corrective rag", "explain reflection in agents")

    def __init__(self, corpus_dir: str | Path, threshold: float = 0.16) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.threshold = threshold
        self._index: VectorIndex | None = None

    @property
    def index(self) -> VectorIndex:
        if self._index is None:
            self._index = VectorIndex.from_directory(self.corpus_dir)
        return self._index

    def run(self, query: str = "", k: int = 3, **_) -> ToolResult:
        if not query.strip():
            raise ToolError("empty query", kind="invalid_input", tool=self.name)

        hits = self.index.search(query, k=k)
        if not hits:
            raise ToolError(
                f"knowledge base returned nothing for {query!r}",
                kind="not_found",
                hint="fall back to web_search or tell the user it is out of scope",
                tool=self.name,
            )

        top = hits[0].score
        grade = "strong" if top >= self.threshold else "weak"

        return ToolResult(
            summary=(
                f"{len(hits)} passage(s), top score {top:.3f} -> {grade} "
                f"(from {hits[0].passage.doc_id})"
            ),
            data={
                "passages": [
                    {
                        "citation": h.passage.citation,
                        "score": h.score,
                        "text": h.passage.text,
                        "preview": h.passage.preview(200),
                    }
                    for h in hits
                ],
                "top_score": top,
            },
            citations=[h.passage.citation for h in hits],
            grade=grade,
            meta={"threshold": self.threshold},
        )
