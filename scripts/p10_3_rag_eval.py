"""P10-3 RAG retrieval evaluation: legacy v0.1 index vs structured BM25.

Dataset note: the 15 answerable queries were authored together with the P10-3
knowledge fragments (development set), so this measures retrieval plumbing and
category routing, not independent industrial generalization. The 3 no-answer
queries check the relevance floor.

Read-only: no network, no simulator.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from printdiag.rag.maintenance import packaged_index  # noqa: E402
from printdiag.rag.structured import _OLD_CATEGORY, packaged_structured_index  # noqa: E402

# (query, expected fault category, acceptable document ids)
EVAL_SET: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("喷嘴温度异常偏高", "喷嘴温度异常", ("p10-3-temperature-nozzle",)),
    ("热床温度上不去", "热床温度异常", ("p10-3-temperature-bed",)),
    ("温度传感器读数跳变", "温度传感器异常",
     ("p10-3-temperature-sensor", "printdiag-thermal-observation")),
    ("固件报热故障，加热元件自检失败", "加热器故障",
     ("p10-3-heater-fault", "printdiag-thermal-observation")),
    ("限位开关一直触发", "限位开关异常",
     ("p10-3-endstop", "printdiag-endstop-observation")),
    ("运动队列一直非空闲", "运动队列异常",
     ("p10-3-motion-queue", "printdiag-device-observation")),
    ("电机丢步或皮带松动", "电机与运动系统异常", ("p10-3-motion-motor",)),
    ("挤出不足，线条变细", "欠挤出", ("p10-3-under-extrusion",)),
    ("喷嘴堵头不出料", "喷嘴堵塞", ("p10-3-nozzle-clog",)),
    ("打印件翘边", "打印件翘边", ("p10-3-warping",)),
    ("首层不粘平台", "首层附着异常", ("p10-3-first-layer",)),
    ("打印中途停止，设备离线", "设备无法启动或打印中断",
     ("p10-3-printer-unavailable", "printdiag-device-observation")),
    ("证据不足怎么办", "证据不足或正常状态", ("p10-3-insufficient-evidence",)),
    ("热故障计数非零", "加热器故障", ("p10-3-heater-fault",)),
    ("x轴限位有效触发但几何未触发", "限位开关异常",
     ("p10-3-endstop", "printdiag-endstop-observation")),
)
NO_ANSWER = ("北京今天天气怎么样", "股票行情预测", "如何烘焙蛋糕")


def _legacy_category(document_id: str) -> str:
    return _OLD_CATEGORY.get(document_id, "证据不足或正常状态")


def evaluate_structured(index) -> dict:
    answerable = []
    for query, category, acceptable in EVAL_SET:
        hits = index.search(query, 3, fault_category=category)
        ids = [hit.document_id for hit in hits]
        rank = next((pos for pos, doc in enumerate(ids, 1) if doc in acceptable), None)
        top = hits[0] if hits else None
        answerable.append({
            "query": query,
            "expected_category": category,
            "acceptable": list(acceptable),
            "returned": [{"document_id": h.document_id, "chunk_id": h.chunk_id,
                          "retrieval_score": h.retrieval_score,
                          "source_reference": h.source_reference} for h in hits],
            "hit": rank is not None,
            "precision": (sum(1 for doc in ids if doc in acceptable) / len(ids)) if ids else 0.0,
            "reciprocal_rank": (1.0 / rank) if rank else 0.0,
            "category_relevant": bool(top and _category_of(hits) == category),
            "citations_valid": all(
                h.chunk_id and h.source_reference and h.retrieval_score is not None for h in hits),
        })
    no_answer = []
    for query in NO_ANSWER:
        hits = index.search(query, 3)
        no_answer.append({"query": query, "returned": len(hits),
                          "handled": len(hits) == 0})
    return _summarize(answerable, no_answer)


def evaluate_legacy(index) -> dict:
    answerable = []
    for query, category, acceptable in EVAL_SET:
        hits = index.search(query, 3)
        ids = [hit.document_id for hit in hits]
        rank = next((pos for pos, doc in enumerate(ids, 1) if doc in acceptable), None)
        answerable.append({
            "query": query,
            "expected_category": category,
            "returned": ids,
            "hit": rank is not None,
            "precision": (sum(1 for doc in ids if doc in acceptable) / len(ids)) if ids else 0.0,
            "reciprocal_rank": (1.0 / rank) if rank else 0.0,
            "category_relevant": bool(ids and _legacy_category(ids[0]) == category),
            # The legacy index has no chunk_id/source_reference/score fields.
            "citations_valid": False,
        })
    no_answer = []
    for query in NO_ANSWER:
        hits = index.search(query, 3)
        no_answer.append({"query": query, "returned": len(hits),
                          "handled": len(hits) == 0})
    return _summarize(answerable, no_answer)


def _category_of(hits) -> str | None:
    if not hits:
        return None
    from printdiag.rag.structured import load_builtin_chunks
    mapping = {chunk.document_id: chunk.fault_category for chunk in load_builtin_chunks()}
    return mapping.get(hits[0].document_id)


def _summarize(answerable: list[dict], no_answer: list[dict]) -> dict:
    n = len(answerable)
    hits = sum(1 for row in answerable if row["hit"])
    precision = sum(row["precision"] for row in answerable) / n
    mrr = sum(row["reciprocal_rank"] for row in answerable) / n
    category = sum(1 for row in answerable if row["category_relevant"])
    citations = sum(1 for row in answerable if row["citations_valid"])
    handled = sum(1 for row in no_answer if row["handled"])
    return {
        "queries": {"answerable": n, "no_answer": len(no_answer)},
        "recall_at_3": {"numerator": hits, "denominator": n},
        "precision_at_3": {"value": round(precision, 4), "denominator": n,
                           "definition": "acceptable hits / returned hits, averaged"},
        "mrr": {"value": round(mrr, 4), "denominator": n},
        "category_relevance": {"numerator": category, "denominator": n},
        "citation_validity_rate": {"numerator": citations, "denominator": n},
        "no_answer_handling_rate": {"numerator": handled, "denominator": len(no_answer)},
        "per_query": answerable,
        "no_answer_cases": no_answer,
    }


def main() -> int:
    structured = evaluate_structured(packaged_structured_index())
    legacy = evaluate_legacy(packaged_index())
    payload = {
        "layer": "offline retrieval evaluation (no network, no simulator)",
        "utc": datetime.now(UTC).isoformat(),
        "dataset_note": "development set authored with the KB; not independent generalization",
        "evaluation_set": {"answerable": len(EVAL_SET), "no_answer": len(NO_ANSWER)},
        "structured_bm25": structured,
        "legacy_v01": legacy,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "runs" / "p10_3_rag_eval" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("structured_bm25", "legacy_v01"):
        m = payload[name]
        print(name, "recall@3", m["recall_at_3"], "mrr", m["mrr"]["value"],
              "category", m["category_relevance"], "citations", m["citation_validity_rate"],
              "no_answer", m["no_answer_handling_rate"])
    print(f"evidence: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
