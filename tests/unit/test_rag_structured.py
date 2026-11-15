"""P10-3 RAG tests: structured BM25 retrieval, citations, no-answer handling."""

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from printdiag.rag.maintenance import packaged_index  # noqa: E402
from printdiag.rag.structured import (  # noqa: E402
    StructuredIndex, load_builtin_chunks, packaged_structured_index,
)

EVAL_PATH = ROOT / "scripts" / "p10_3_rag_eval.py"
_spec = importlib.util.spec_from_file_location("p10_3_rag_eval", EVAL_PATH)
rag_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rag_eval)


@pytest.fixture(scope="module")
def index():
    return packaged_structured_index()


def test_index_size_and_categories(index):
    assert index.available and index.size == 16
    assert len(index.categories()) == 13


@pytest.mark.parametrize(("query", "category", "expected"), rag_eval.EVAL_SET)
def test_answerable_queries_hit_expected_documents(index, query, category, expected):
    hits = index.search(query, 3, fault_category=category)
    ids = [hit.document_id for hit in hits]
    assert any(doc in ids for doc in expected), (query, ids)
    top = hits[0]
    assert top.chunk_id and top.source_reference and top.retrieval_score is not None


@pytest.mark.parametrize("query", rag_eval.NO_ANSWER)
def test_no_answer_queries_return_nothing(index, query):
    assert index.search(query, 3) == []


def test_category_filter_changes_ranking(index):
    plain = index.search("温度异常 热故障", 3)
    filtered = index.search("温度异常 热故障", 3, fault_category="加热器故障")
    assert filtered and filtered[0].document_id == "p10-3-heater-fault"
    assert plain


def test_empty_index_degrades():
    empty = StructuredIndex([])
    assert empty.available is False and empty.size == 0
    assert empty.search("喷嘴温度", 3) == []


def test_legacy_index_has_no_citation_metadata():
    hits = packaged_index().search("限位开关", 3)
    assert hits
    assert all(hit.chunk_id is None and hit.source_reference is None for hit in hits)


def test_builtin_chunks_are_unique_and_complete():
    chunks = load_builtin_chunks()
    keys = {(chunk.document_id, chunk.chunk_id) for chunk in chunks}
    assert len(keys) == len(chunks) == 16
    for chunk in chunks:
        assert chunk.source_reference
        assert chunk.license_basis
        assert chunk.excerpt


def test_retrieval_evaluation_metrics():
    structured = rag_eval.evaluate_structured(packaged_structured_index())
    legacy = rag_eval.evaluate_legacy(packaged_index())
    assert structured["recall_at_3"] == {"numerator": 15, "denominator": 15}
    assert structured["mrr"]["value"] == 1.0
    assert structured["category_relevance"] == {"numerator": 15, "denominator": 15}
    assert structured["citation_validity_rate"] == {"numerator": 15, "denominator": 15}
    assert structured["no_answer_handling_rate"] == {"numerator": 3, "denominator": 3}
    assert legacy["recall_at_3"]["numerator"] < structured["recall_at_3"]["numerator"]
    assert legacy["citation_validity_rate"]["numerator"] == 0
