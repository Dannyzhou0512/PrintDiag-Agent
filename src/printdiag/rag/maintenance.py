"""Search one packaged, reviewed maintenance collection with no path input."""

from collections import Counter
from importlib.resources import files
from pathlib import Path
import re
from typing import Annotated

from pydantic import Field, TypeAdapter, ValidationError

from printdiag.schemas.agent import KnowledgeHit, ShortText

_COLLECTION = ("knowledge", "maintenance", "printdiag-v01.json")
_TERMS = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*|[\u3400-\u9fff]+")


class MaintenanceFragment(KnowledgeHit):
    source: ShortText
    applicable_version: ShortText
    license_basis: ShortText
    heading: ShortText


_CORPUS = TypeAdapter(Annotated[list[MaintenanceFragment], Field(min_length=1, max_length=20)])


def _terms(text: str) -> set[str]:
    terms = set()
    for word in _TERMS.findall(text.casefold()):
        if word.isascii():
            terms.add(word)
        else:
            terms.update(word[i:i + 2] for i in range(len(word) - 1))
    return terms


class MaintenanceIndex:
    def __init__(self, fragments: tuple[MaintenanceFragment, ...] = ()) -> None:
        keys = {(fragment.document_id, fragment.anchor, fragment.revision) for fragment in fragments}
        if len(keys) != len(fragments):
            raise ValueError("maintenance fragment identities must be unique")
        self._fragments = fragments
        self._terms = tuple(_terms(f"{item.title} {item.heading} {item.excerpt}") for item in fragments)
        self._frequency = Counter(term for terms in self._terms for term in terms)

    @property
    def available(self) -> bool:
        return bool(self._fragments)

    def search(self, query: str, limit: int) -> list[KnowledgeHit]:
        wanted = _terms(query)
        ranked = []
        for item, terms in zip(self._fragments, self._terms, strict=True):
            score = sum(1 / self._frequency[term] for term in sorted(wanted & terms))
            if score:
                ranked.append((-score, item.document_id, item.anchor, item))
        ranked.sort(key=lambda row: row[:3])
        return [KnowledgeHit(**item.model_dump(include=set(KnowledgeHit.model_fields)))
                for _, _, _, item in ranked[:limit]]


def packaged_index() -> MaintenanceIndex:
    """Read only the shipped resource, rejecting linked files and directories."""
    resource = files("printdiag.rag")
    try:
        for name in _COLLECTION:
            resource = resource.joinpath(name)
            if isinstance(resource, Path) and (resource.is_symlink() or resource.is_junction()):
                return MaintenanceIndex()
        with resource.open("rb") as stream:
            payload = stream.read(65537)
        if len(payload) > 65536:
            return MaintenanceIndex()
        return MaintenanceIndex(tuple(_CORPUS.validate_json(payload)))
    except (OSError, ValidationError, ValueError):
        return MaintenanceIndex()
