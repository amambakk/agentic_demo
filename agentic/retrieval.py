"""
A tiny retrieval engine: tokenizer, TF-IDF vectoriser, cosine ranking.

Teaching point
--------------
People assume "RAG" requires an embedding model. It does not require one to
*teach* the idea. Retrieval is: turn text into vectors, score the query
against every passage, return the top k. Swapping TF-IDF for embeddings
changes the quality of the scores, not the shape of the pipeline -- so the
architecture you learn here is the architecture you keep.

Everything below is the standard library: `re`, `math`, `collections`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9']+")

# A small stop list. Real systems use a larger one, or rely on IDF to
# suppress common words automatically (which it mostly does).
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "no", "not", "of", "on", "or", "over", "so", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "to", "up", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your",
}


# Longest suffixes first. This is a crude stemmer, not Porter.
_SUFFIXES = (
    "ational", "ations", "ation", "ings", "ions", "ing", "ion", "edly", "ies", "ily",
    "ment", "ness", "ity", "ive", "ate", "ed", "ly", "es", "al", "s", "y", "e",
)


def stem(word: str) -> str:
    """Collapse inflections so 'evaluate', 'evaluation' and 'evaluating' match.

    Teaching point -- lexical mismatch
    ----------------------------------
    Bag-of-words retrieval scores exact token overlap, so 'evaluate' and
    'evaluation' are, to TF-IDF, entirely unrelated words. That single fact
    causes a surprising share of "the retriever missed the obvious document"
    complaints. Stemming is the cheap partial fix. Dense embeddings are the
    real fix, because they collapse *semantic* distance rather than just
    spelling -- which is the actual argument for using them, rather than
    "embeddings are more modern".

    Two passes, because stripping one suffix can expose another.
    """
    for _ in range(2):
        if len(word) <= 4:
            return word
        for suffix in _SUFFIXES:
            if not word.endswith(suffix):
                continue
            # Never strip a plural 's' off a word ending in 'ss' (process).
            if suffix == "s" and word.endswith("ss"):
                continue
            if len(word) - len(suffix) < 4:
                continue
            word = word[: -len(suffix)]
            break
        else:
            return word
    return word


def tokenize(text: str, drop_stopwords: bool = True, apply_stem: bool = True) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if apply_stem:
        tokens = [stem(t) for t in tokens]
    return tokens


@dataclass
class Passage:
    doc_id: str
    chunk_id: int
    text: str
    meta: dict = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"[{self.doc_id}#p{self.chunk_id}]"

    def preview(self, width: int = 160) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= width else flat[: width - 3] + "..."


@dataclass
class Hit:
    passage: Passage
    score: float


def chunk_document(doc_id: str, text: str, meta: dict | None = None) -> list[Passage]:
    """Split on blank lines -- paragraph-level chunks.

    Chunk size is a real design lever: too large and the query signal is
    diluted across unrelated sentences, too small and the passage loses the
    context needed to interpret it.
    """
    meta = meta or {}
    passages: list[Passage] = []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    for i, block in enumerate(blocks):
        if i == 0:
            # Skip a leading markdown title or a `KEY: value` header block --
            # they are metadata, not content. Indexing them gives you a
            # passage whose "answer" is the word DATE, which is the kind of
            # junk chunk that quietly drags down retrieval quality.
            if block.startswith("#"):
                meta = {**meta, "title": block.lstrip("# ").strip()}
                continue
            if all(re.match(r"^[A-Z]+:\s", line) for line in block.splitlines()):
                continue
        passages.append(
            Passage(
                doc_id=doc_id,
                chunk_id=len(passages),
                text=block,
                meta=meta,
                tokens=tokenize(block),
            )
        )
    return passages


class VectorIndex:
    """TF-IDF index with cosine similarity search."""

    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self.n = len(passages) or 1
        self.df: Counter[str] = Counter()
        for p in passages:
            for term in set(p.tokens):
                self.df[term] += 1
        self._vectors = [self._vectorize(p.tokens) for p in passages]

    # -- vector maths -----------------------------------------------------

    def idf(self, term: str) -> float:
        # Smoothed IDF: rare terms score high, ubiquitous terms score ~1.
        return math.log((self.n + 1) / (self.df.get(term, 0) + 1)) + 1.0

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        longest = max(tf.values())
        vec = {t: (c / longest) * self.idf(t) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        # Vectors are pre-normalised, so the dot product *is* the cosine.
        if len(a) > len(b):
            a, b = b, a
        return sum(w * b.get(t, 0.0) for t, w in a.items())

    # -- search -----------------------------------------------------------

    def search(self, query: str, k: int = 3) -> list[Hit]:
        qvec = self._vectorize(tokenize(query))
        if not qvec:
            return []
        scored = [
            Hit(passage=p, score=round(self._cosine(qvec, v), 4))
            for p, v in zip(self.passages, self._vectors)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return [h for h in scored[:k] if h.score > 0]

    # -- construction helpers --------------------------------------------

    @classmethod
    def from_directory(cls, directory: str | Path, pattern: str = "*.txt") -> "VectorIndex":
        directory = Path(directory)
        passages: list[Passage] = []
        for path in sorted(directory.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            meta = _parse_headers(text)
            meta["path"] = str(path)
            passages.extend(chunk_document(path.name, text, meta))
        return cls(passages)

    def stats(self) -> dict:
        return {
            "passages": len(self.passages),
            "documents": len({p.doc_id for p in self.passages}),
            "vocabulary": len(self.df),
        }


def _parse_headers(text: str) -> dict:
    """Read simple `KEY: value` headers off the top of a document."""
    meta: dict = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Z]+):\s*(.+)$", line.strip())
        if not m:
            break
        meta[m.group(1).lower()] = m.group(2).strip()
    return meta
