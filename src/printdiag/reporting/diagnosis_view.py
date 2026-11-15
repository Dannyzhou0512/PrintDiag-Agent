"""Allowlisted Console projection of already validated diagnosis data."""

from __future__ import annotations

from html import escape
import re

from printdiag.diagnostics.fault_categories import CANONICAL_CATEGORIES, INSUFFICIENT_CATEGORY
from printdiag.reporting.safety_review import review_text

_STATUS = {
    "complete": "报告完成",
    "partial": "部分支持",
    "insufficient_evidence": "证据不足",
    "failed": "诊断失败",
}
_SAFETY = {"blocked": "危险建议已拦截", "review_required": "需人工复核",
           "clear": "未检出风险", "not_evaluated": "未评估"}
_SOURCE = {"LIVE_MODEL": "Live 模型", "SAVED_REPLAY": "保存回放",
           "MOCK_TEST": "Mock 测试", "DEVICE_ONLY": "仅设备读取"}


def _text(value, limit=500):
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _safe_summary(value, limit=500):
    return _text(value, limit) if isinstance(value, str) and not review_text(value) else ""


def _objects(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def safety_status(safety):
    if not isinstance(safety, dict):
        return "not_evaluated"
    if safety.get("status") == "review_required":
        return "review_required"
    if safety.get("blocked") is True:
        return "blocked"
    return "clear" if safety.get("status") == "clear" and safety.get("blocked") is False else "not_evaluated"


def canonical_category(value):
    return value if isinstance(value, str) and value != INSUFFICIENT_CATEGORY and value in CANONICAL_CATEGORIES else None


def make_view(report, meta, *, question="", presentation=None, historical=False):
    """Never derive device support from prose, draft observations or RAG matches."""
    status = report.get("status")
    safety = safety_status(meta.get("safety"))
    guard = (meta.get("guard") or {}).get("valid")
    clear = safety == "clear" and (guard is True or historical and guard is None)
    presentation = presentation if clear and isinstance(presentation, dict) else {}
    category = canonical_category(presentation.get("category"))
    decision = meta.get("v2_decision")
    title = f"疑似{category}" if category else "打印问题排查"
    if status == "failed":
        title = "诊断未完成"
    elif safety == "blocked":
        title = "危险建议已拦截"
    elif safety == "review_required":
        title = "建议需人工复核"
    elif guard is False:
        title = "本次未生成可用建议"
    elif decision == "contradicted":
        title = "现有观测未支持故障假设"
    notice = None
    if safety == "blocked":
        notice = "安全审查已拦截危险内容，请暂停相关操作。"
    elif safety == "review_required":
        notice = "维修条件尚待确认，请先请专业人员复核。"
    elif clear:
        notice = _safe_summary(presentation.get("safety_note")) or None

    receipts = [item for item in _objects(meta.get("device_evidence"))
                if item.get("evidence_id") and item.get("observed_at_utc")
                and item.get("tool") in ("read_printer_state", "read_endstop_components")]
    device_source = meta.get("device_source")
    checks = []
    if status != "failed" and guard is not False and safety not in ("blocked", "review_required"):
        suggestions = presentation.get("checks") or report.get("recommended_checks") or []
        if isinstance(suggestions, list):
            checks = list(dict.fromkeys(text for item in suggestions if (text := _safe_summary(item))))[:4]

    rag = _objects(meta.get("rag_hits"))
    titles = list(dict.fromkeys(text for hit in rag
                               if (text := _safe_summary(hit.get("source_title") or hit.get("title"), 100))))[:2]
    basis = []
    symptom = _safe_summary(question)
    if symptom:
        basis.append({"source": "用户描述", "text": symptom})
    if receipts:
        scope = "模拟器只读状态" if device_source == "LIVE_SIMULATOR" else "只读回执（来源见详情）"
        basis.append({"source": scope,
                      "text": "以下为模拟环境状态，仅供参考。" if device_source == "LIVE_SIMULATOR"
                      else "本轮保存了只读回执摘要。"})
    if device_source == "SYNTHETIC_FIXTURE" and not receipts:
        basis.append({"source": "合成示例", "text": "本轮使用模拟样本，没有采集真实设备回执。"})
    if titles:
        basis.append({"source": "维护知识背景", "text": "、".join(titles)})
    uncertainty = ("本次没有可用于验证该故障的设备证据。" if decision != "supported" or not receipts
                   else "现有回执支持核查该现象，具体原因仍待验证。")
    if status == "failed":
        uncertainty = "本次请求未完成，未产生可用的诊断报告。"
    elif guard is False:
        uncertainty = "模型输出未通过格式校验，请保留描述并稍后重新诊断。"
    elif not category and decision == "insufficient" and not checks:
        uncertainty = "请补充实际观察到的现象、出现条件或报错。"
    legacy = (_safe_summary(report.get("summary")) if historical and not meta.get("guard")
              and safety not in ("blocked", "review_required") else "")
    mode = meta.get("source_mode")
    if not mode:
        mode = ("SAVED_REPLAY" if meta.get("scenario_id") else "LIVE_MODEL" if meta.get("live_inference") is True
                else "MOCK_TEST" if meta.get("mock") is True else None)
    model = {"base": "Base", "lora": "LoRA"}.get(meta.get("model"))
    return {
        "category": category,
        "title": title,
        "status": _STATUS.get(status, "状态未记录"),
        "status_code": status,
        "source": _SOURCE.get(mode, "来源未记录"),
        "model": model,
        "basis": basis[:4],
        "checks": checks,
        "safety": {"label": _SAFETY[safety], "status": safety, "notice": notice},
        "uncertainty": uncertainty,
        "legacy_summary": legacy or None,
        "technical": {
            "source": mode, "model": model, "model_source": _text(meta.get("model_source"), 100),
            "device_source": _text(device_source, 100),
            "references": [{key: _safe_summary(hit.get(key), 180) for key in
                            ("document_id", "chunk_id", "source_title", "source_reference")}
                           for hit in rag[:12]],
            "receipts": [{key: _text(item.get(key), 100) for key in
                          ("tool", "evidence_id", "observed_at_utc")} for item in receipts[:12]],
            "schema_valid": guard, "v2_decision": decision,
            "v2_codes": [_text(code, 100) for code in (meta.get("v2_codes") or [])[:20]],
            "safety": safety,
            "elapsed_ms": meta.get("server_latency_ms", meta.get("latency_ms")),
            "audit": _text(meta.get("audit"), 100),
        },
    }


def history_view(record):
    """Reconstruct a safe report from saved structured fields, including old rows."""
    report = record.get("validated_report") or {"status": record.get("diagnosis_status")}
    answer = record.get("answer") or {}
    saved = answer.get("diagnosis_view")
    if not isinstance(saved, dict) and answer.get("conclusion"):
        report = {**report, "summary": answer["conclusion"]}
    seed = {"category": record.get("fault_category")}
    if isinstance(saved, dict):
        seed["checks"] = saved.get("checks")
        seed["safety_note"] = (saved.get("safety") or {}).get("notice")
    guard = (saved.get("technical") or {}).get("schema_valid") if isinstance(saved, dict) else None
    return make_view(report, {
        "source_mode": record.get("source_mode"), "model": record.get("model_variant"),
        "model_source": record.get("model_name"), "safety": record.get("safety_result"),
        "guard": {"valid": guard} if guard is not None else None,
        "device_source": record.get("device_source"), "device_evidence": record.get("evidence_summary"),
        "rag_hits": record.get("rag_references"), "v2_decision": record.get("v2_decision"),
        "v2_codes": record.get("v2_codes"), "latency_ms": record.get("duration_ms"),
        "audit": "历史记录已保存",
    }, question=record.get("user_question"), presentation=seed, historical=True)


def _markdown_text(value):
    # Treat every dynamic field as inert text, including Markdown links and HTML.
    return re.sub(r"([\\`*{}\[\]<>()#.!|~:+])", r"\\\1", escape(_text(value, 1000), quote=False))


def view_markdown(view, *, heading="# PrintDiag 诊断报告"):
    text = _markdown_text
    tech = view["technical"]
    lines = [heading, "", f"## {text(view['title'])}", text(view["status"]), ""]
    if view["legacy_summary"]:
        lines.extend((text(view["legacy_summary"]), ""))
    lines += ["## 判断依据", *[f"- {text(item['source'])}：{text(item['text'])}" for item in view["basis"]]]
    lines += ["", "## 建议检查", *[f"{i}. {text(item)}" for i, item in enumerate(view["checks"], 1)]]
    if not view["checks"]:
        lines.append("暂无可发布的检查建议。")
    lines += ["", "## 安全提醒", text(view["safety"]["notice"] or view["safety"]["label"]),
              "", "## 尚需确认", text(view["uncertainty"]),
              "", "<details>", "<summary>技术详情</summary>", "",
              f"- 状态：{text(view['status_code'])}",
              f"- 模型与来源：{text(view['model'] or '')} {text(tech['source'] or '')} {text(tech['model_source'])}",
              f"- 设备来源：{text(tech['device_source'])}",
              f"- V2：{text(tech['v2_decision'] or '')} {text(' / '.join(tech['v2_codes']))}",
              f"- Safety：{text(tech['safety'])}",
              f"- 七字段校验：{'通过' if tech['schema_valid'] is True else '未通过' if tech['schema_valid'] is False else '未记录'}",
              f"- 处理耗时：{text(str(tech['elapsed_ms'])) + ' ms' if tech['elapsed_ms'] is not None else '未记录'}",
              f"- 审计记录：{text(tech['audit']) or '未记录'}",
              "", "### RAG 来源"]
    lines += [f"- {text(r['document_id'])} / {text(r['chunk_id'])} / {text(r['source_reference'])}"
              for r in tech["references"]] or ["- 未执行知识检索，或本轮无可用引用。"]
    lines += ["", "### 设备证据"]
    lines += [f"- {text(r['tool'])} / {text(r['evidence_id'])} / {text(r['observed_at_utc'])}"
              for r in tech["receipts"]] or ["- 无可引用的回执。"]
    lines += ["", "</details>", ""]
    return "\n".join(lines)
