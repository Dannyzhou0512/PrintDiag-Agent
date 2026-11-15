"""Validation-preserving mapping from a diagnosis result to history."""

from __future__ import annotations

from datetime import UTC, datetime
import os
from typing import Any
from uuid import uuid4

from printdiag.reporting.diagnosis_view import history_view, view_markdown

from printdiag.history.store import (
    HistoryRecordError,
    HistoryStore,
    SaveResult,
    default_history_path,
)


class HistoryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


SOURCE_MODES = {"LIVE_MODEL", "SAVED_REPLAY", "MOCK_TEST", "DEVICE_ONLY"}
DIAGNOSIS_STATUSES = {"complete", "partial", "insufficient_evidence", "failed"}
_SAFE_EVIDENCE_KEYS = ("tool", "evidence_id", "observed_at_utc", "data")
_SAFE_RAG_KEYS = (
    "document_id", "chunk_id", "source_title", "source_type",
    "retrieval_score", "retrieval_rank", "knowledge_version",
    "source_reference", "excerpt",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _clean_evidence(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items[:30]:
        if not isinstance(item, dict):
            continue
        clean = {key: item[key] for key in _SAFE_EVIDENCE_KEYS if key in item}
        if clean:
            cleaned.append(clean)
    return cleaned


def _clean_rag(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    cleaned = []
    for rank, item in enumerate(items[:20], start=1):
        if not isinstance(item, dict):
            continue
        clean = {key: item[key] for key in _SAFE_RAG_KEYS if key in item}
        if clean.get("document_id"):
            clean.setdefault("source_type", "curated_knowledge")
            clean.setdefault("retrieval_rank", rank)
            cleaned.append(clean)
    return cleaned


def _source_mode(meta: dict[str, Any], requested: str | None) -> str:
    if requested:
        if requested not in SOURCE_MODES:
            raise HistoryError("INVALID_SOURCE_MODE", f"unsupported source mode: {requested}")
        return requested
    if meta.get("scenario_id") or meta.get("model_source") == "SAVED_MODEL_REPLAY":
        return "SAVED_REPLAY"
    if meta.get("live_inference") is True:
        return "LIVE_MODEL"
    if meta.get("mock") is True or str(meta.get("mode", "")).startswith("MOCK"):
        return "MOCK_TEST"
    return "DEVICE_ONLY"


def _model_identity(meta: dict[str, Any]) -> tuple[str | None, str | None]:
    variant = meta.get("model")
    if variant not in ("base", "lora"):
        variant = None
    name = meta.get("model_name") or meta.get("requested_model") or "Qwen3.5-4B"
    return name, variant


class HistoryService:
    def __init__(self, store: HistoryStore, *, retention_days: int = 365) -> None:
        self.store = store
        self.retention_days = max(0, int(retention_days))

    @classmethod
    def from_environment(cls) -> "HistoryService":
        return cls(
            HistoryStore(default_history_path()),
            retention_days=int(os.environ.get("PRINTDIAG_HISTORY_RETENTION_DAYS", "365")),
        )

    def save_result(self, result: dict[str, Any], *, question: str, intent: str | None,
                    source_mode: str | None = None, request_id: str | None = None,
                    completed_at: str | None = None) -> SaveResult:
        """Persist a result only after the caller has a validated report."""
        if not isinstance(result, dict):
            raise HistoryError("INVALID_RESULT", "result is not an object")
        report = result.get("report")
        meta = result.get("meta") or {}
        answer = result.get("answer")
        if not isinstance(report, dict):
            raise HistoryError("REPORT_REQUIRED", "only validated report results can be saved")
        if not isinstance(meta, dict):
            meta = {}
        if not question or not str(question).strip():
            raise HistoryError("QUESTION_REQUIRED", "question is required")
        diagnosis_status = report.get("status")
        if diagnosis_status not in DIAGNOSIS_STATUSES - {"failed"}:
            raise HistoryError("INVALID_REPORT_STATUS", f"unexpected report status: {diagnosis_status}")
        diagnosis_id = report.get("diagnosis_id")
        if not diagnosis_id or not isinstance(diagnosis_id, str):
            raise HistoryError("DIAGNOSIS_ID_REQUIRED", "validated report has no diagnosis_id")
        mode = _source_mode(meta, source_mode)
        request_id = (request_id or meta.get("request_id") or uuid4().hex)
        now = _now()
        model_name, model_variant = _model_identity(meta)
        evidence = _clean_evidence(meta.get("device_evidence"))
        rag = _clean_rag(meta.get("rag_hits"))
        evidence_refs = report.get("evidence_refs") if isinstance(report.get("evidence_refs"), list) else []
        record = {
            "diagnosis_id": diagnosis_id,
            "request_id": str(request_id),
            "created_at": now,
            "completed_at": completed_at or now,
            "user_question": str(question).strip(),
            "intent": intent,
            "source_mode": mode,
            "model_name": model_name,
            "model_variant": model_variant,
            "diagnosis_status": diagnosis_status,
            "fault_category": meta.get("fault_category"),
            "validated_report": report,
            "evidence_summary": evidence,
            "rag_references": rag,
            "v03_validation": {
                "valid": True,
                "status": diagnosis_status,
                "hypothesis_count": len(report.get("hypotheses") or []),
                "evidence_ref_count": len(evidence_refs),
                "limitation_count": len(report.get("limitations") or []),
            },
            "v2_decision": meta.get("v2_decision"),
            "v2_codes": meta.get("v2_codes") or [],
            "safety_result": meta.get("safety"),
            "device_source": meta.get("device_source") or meta.get("sources"),
            "device_snapshot_reference": {
                "evidence_ids": [item.get("evidence_id") for item in evidence
                                 if item.get("evidence_id")],
                "observed_at_utc": [item.get("observed_at_utc") for item in evidence
                                    if item.get("observed_at_utc")],
            },
            "answer": answer if isinstance(answer, dict) else None,
            "duration_ms": meta.get("server_latency_ms") or meta.get("latency_ms"),
            "error_code": None,
            "schema_version": 1,
            "persistence_status": "saved",
        }
        return self.store.save(record)

    def save_failure(self, *, question: str, intent: str | None, source_mode: str,
                     error_code: str, request_id: str | None = None,
                     completed_at: str | None = None) -> SaveResult:
        """Persist limited failure metadata; never mark it as a success."""
        if source_mode not in SOURCE_MODES:
            raise HistoryError("INVALID_SOURCE_MODE", f"unsupported source mode: {source_mode}")
        request_id = request_id or uuid4().hex
        now = _now()
        record = {
            "diagnosis_id": f"failed-{request_id}",
            "request_id": str(request_id),
            "created_at": now,
            "completed_at": completed_at or now,
            "user_question": str(question).strip()[:500] or "(empty)",
            "intent": intent,
            "source_mode": source_mode,
            "model_name": None,
            "model_variant": None,
            "diagnosis_status": "failed",
            "fault_category": None,
            "validated_report": None,
            "evidence_summary": None,
            "rag_references": None,
            "v03_validation": None,
            "v2_decision": None,
            "v2_codes": None,
            "safety_result": None,
            "device_source": None,
            "device_snapshot_reference": None,
            "answer": None,
            "duration_ms": None,
            "error_code": str(error_code)[:80],
            "schema_version": 1,
            "persistence_status": "saved",
        }
        return self.store.save(record)

    def list(self, **kwargs: Any) -> dict[str, Any]:
        return self.store.query(**kwargs)

    def get(self, diagnosis_id: str) -> dict[str, Any] | None:
        return self.store.get(diagnosis_id)

    def report_markdown(self, diagnosis_id: str) -> str | None:
        record = self.get(diagnosis_id)
        return render_markdown(record) if record else None

    def prune(self, *, now: datetime | None = None) -> int:
        return self.store.prune(retention_days=self.retention_days, now=now)


def render_markdown(record: dict[str, Any]) -> str:
    """Render the saved public fields without rerunning the diagnosis."""
    return (view_markdown(history_view(record), heading="# PrintDiag 诊断记录")
            + "\n> 报告来自已保存的诊断记录，不重新调用模型。\n")
