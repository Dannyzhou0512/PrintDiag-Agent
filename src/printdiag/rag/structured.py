"""Structured maintenance knowledge with auditable BM25 sparse retrieval.

This is a sparse keyword/BM25 retriever over curated project knowledge, with
category-aware ranking and per-hit provenance (document_id, chunk_id,
source_reference, retrieval_score). It is **not** an embedding/vector RAG and
does not claim semantic search.

Knowledge text is untrusted reference material: it can explain and suggest
checks, but it is never device evidence and never controls tool calls.
"""

from collections import Counter
import json
import math
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from printdiag.schemas.agent import KnowledgeHit, ShortText

_OLD_COLLECTION = ("knowledge", "maintenance", "printdiag-v01.json")
_NEW_COLLECTION = ("knowledge", "maintenance", "printdiag-p10_3.jsonl")
_TERMS = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*|[\u3400-\u9fff]+")
_LICENSE_BASIS = "project-internal curated knowledge"

_OLD_CATEGORY = {
    "printdiag-endstop-observation": "限位开关异常",
    "printdiag-device-observation": "运动队列异常",
    "printdiag-thermal-observation": "温度传感器异常",
}

FaultCategory = str


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: ShortText
    chunk_id: ShortText
    title: ShortText
    source: ShortText
    version: ShortText
    fault_category: ShortText
    applicable_device: ShortText
    symptoms: list[str] = Field(default_factory=list)
    possible_causes: list[str] = Field(default_factory=list)
    recommended_checks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_reference: ShortText
    anchor: ShortText
    revision: str
    license_basis: ShortText
    excerpt: ShortText

    def text(self) -> str:
        return " ".join([
            self.title, self.fault_category, self.applicable_device,
            " ".join(self.symptoms), " ".join(self.possible_causes),
            " ".join(self.recommended_checks), " ".join(self.limitations), self.excerpt,
        ])


def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for word in _TERMS.findall(text.casefold()):
        if word.isascii():
            terms.append(word)
        else:
            terms.extend(word)
            terms.extend(word[i:i + 2] for i in range(len(word) - 1))
    return terms


def _resource_text(parts: tuple[str, ...]) -> str | None:
    base = Path(__file__).parent / "knowledge" / "maintenance"
    path = base.joinpath(*parts[2:])
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def load_builtin_chunks() -> list[KnowledgeChunk]:
    """Load the v0.1 corpus plus the P10-3 structured fragments (read-only)."""
    chunks: list[KnowledgeChunk] = []
    old = _resource_text(_OLD_COLLECTION)
    if old:
        for row in json.loads(old):
            chunks.append(KnowledgeChunk(
                document_id=row["document_id"],
                chunk_id=row["anchor"],
                title=row["title"],
                source=row["source"],
                version=row.get("applicable_version", "v0.1"),
                fault_category=_OLD_CATEGORY.get(row["document_id"], "证据不足或正常状态"),
                applicable_device="Marlin Simulator diagnostic projection (M1/M2)",
                symptoms=[row.get("heading", row["title"])],
                possible_causes=[],
                recommended_checks=[],
                limitations=[],
                source_reference=row["source"],
                anchor=row["anchor"],
                revision=row["revision"],
                license_basis=row.get("license_basis", _LICENSE_BASIS),
                excerpt=row["excerpt"],
            ))
    new = _resource_text(_NEW_COLLECTION)
    if new:
        for line in new.splitlines():
            if line.strip():
                chunks.append(KnowledgeChunk.model_validate_json(line))
    return chunks


class StructuredIndex:
    """BM25 index over curated maintenance chunks with category-aware ranking."""

    def __init__(self, chunks: list[KnowledgeChunk], *, k1: float = 1.5, b: float = 0.75,
                 min_score: float = 4.0) -> None:
        unique = {(chunk.document_id, chunk.chunk_id) for chunk in chunks}
        if len(unique) != len(chunks):
            raise ValueError("knowledge chunk identities must be unique")
        self._chunks = tuple(chunks)
        self._k1 = k1
        self._b = b
        # Deterministic relevance floor: keeps "no answer" queries from returning
        # coincidental single-character overlaps. Chosen from the retrieval
        # evaluation set (relevant hits score >= ~9, unrelated <= ~3).
        self._min_score = min_score
        self._tokens = tuple(Counter(_terms(chunk.text())) for chunk in chunks)
        self._lengths = tuple(sum(counter.values()) for counter in self._tokens)
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._df = Counter(term for counter in self._tokens for term in counter)
        total = len(self._chunks)
        self._idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5)) for term, df in self._df.items()
        }

    @property
    def available(self) -> bool:
        return bool(self._chunks)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({chunk.fault_category for chunk in self._chunks}))

    def _score(self, query_terms: list[str], index: int, chunk: KnowledgeChunk,
               category: str | None) -> float:
        counts = self._tokens[index]
        length = self._lengths[index] or 1
        score = 0.0
        for term in dict.fromkeys(query_terms):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = self._idf.get(term, 0.0)
            denominator = frequency + self._k1 * (1 - self._b + self._b * length / (self._avg_len or 1))
            score += idf * frequency * (self._k1 + 1) / denominator
        if category is not None and chunk.fault_category == category:
            score += 2.0  # category routing bonus, documented and deterministic
        return score

    def search(self, query: str, limit: int = 3, *, fault_category: str | None = None) -> list[KnowledgeHit]:
        terms = _terms(query)
        if not terms or limit < 1:
            return []
        scored = []
        for index, chunk in enumerate(self._chunks):
            score = self._score(terms, index, chunk, fault_category)
            if score >= self._min_score:
                scored.append((-score, chunk.document_id, chunk.chunk_id, chunk))
        scored.sort(key=lambda row: row[:3])
        hits = []
        for negative, _, _, chunk in scored[:limit]:
            hits.append(KnowledgeHit(
                document_id=chunk.document_id,
                title=chunk.title,
                anchor=chunk.anchor,
                revision="p10.3" if chunk.revision.startswith("p10") else chunk.revision,
                excerpt=chunk.excerpt,
                chunk_id=chunk.chunk_id,
                source_reference=chunk.source_reference,
                retrieval_score=round(-negative, 6),
            ))
        return hits

    def search_with_category(self, query: str, limit: int, category: str) -> list[KnowledgeHit]:
        return self.search(query, limit, fault_category=category)


def packaged_structured_index() -> StructuredIndex:
    """Read-only packaged index; an unreadable collection degrades to empty."""
    try:
        return StructuredIndex(load_builtin_chunks())
    except (OSError, ValueError, ValidationError):
        return StructuredIndex([])
