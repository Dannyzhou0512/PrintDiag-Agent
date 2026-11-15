"""V2 taxonomy audit: the frozen V0.3 37-label vocabulary is not modified."""

import hashlib
from typing import Iterable

from printdiag.diagnostics.fault_categories import (
    CANONICAL_CATEGORIES, normalize_fault_category,
)

# Candidate V2-only extensions; must be validated on train/val or domain rules
# before any adoption. Never applied to V0.3.
V2_EXTENDED_CATEGORIES = ("模型网格错误", "支撑结构失效")


def audit_taxonomy(categories: Iterable[str]) -> dict:
    canonical = sorted(CANONICAL_CATEGORIES)
    mappings = []
    missing = []
    for category in sorted(set(categories)):
        match = normalize_fault_category(category)
        mappings.append({"label": category, "canonical": match.canonical, "match": match.match})
        if match.canonical is None:
            missing.append(category)
    return {
        "canonical_count": len(canonical),
        "canonical_sha256": hashlib.sha256(
            "\n".join(canonical).encode("utf-8")).hexdigest(),
        "canonical": canonical,
        "input_categories": sorted(set(categories)),
        "mappings": mappings,
        "unmapped_categories": missing,
        "v2_extended_candidates": list(V2_EXTENDED_CATEGORIES),
        "v2_extension_note": "candidates only; not part of the frozen V0.3 taxonomy and not "
                             "applied to historical predictions",
    }
