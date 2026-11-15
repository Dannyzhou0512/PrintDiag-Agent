"""Adapt a validated seven-field SFT draft onto the existing Agent report contract.

The LoRA path is deliberately separate from ``ReportDraft``: the model never
supplies evidence references, so the adapter projects this diagnosis's
executor-issued ``ToolResult`` objects into receipts, verifies them with
``verify_evidence`` and then reuses ``build_report`` for the public contract.
"""

from collections.abc import Sequence

from pydantic import ValidationError

from printdiag.diagnostics.evidence_verifier import ToolExecutionEvidence, verify_evidence
from printdiag.diagnostics.fault_categories import normalize_fault_category
from printdiag.reporting.agent_report import Check, ReportDraft, build_report
from printdiag.reporting.safety_review import review_draft, safe_checks
from printdiag.schemas.agent import (
    DiagnosisReport, EndstopData, ErrorCode, EvidenceKey, Hypothesis, KnowledgeData, Limitation,
    PrinterStateProjection, ToolResult,
)
from printdiag.schemas.sft_diagnosis import DiagnosisValidationError, validate_diagnosis

_DEVICE_TOOLS = ("read_printer_state", "read_endstop_components")
_MAX_SUMMARY = 490
_MAX_TEXT = 480


def _state_observations(data: PrinterStateProjection) -> tuple[str, ...]:
    facts: list[str] = []
    if data.idle is not None:
        facts.append(f"idle={str(data.idle).lower()}")
    if data.moves_queued is not None:
        facts.append(f"moves_queued={data.moves_queued}")
    if data.timers_active is not None:
        facts.append(f"timers_active={str(data.timers_active).lower()}")
    if data.watchdog_resets is not None:
        facts.append(f"watchdog_resets={data.watchdog_resets}")
    if data.thermal_fault_count is not None:
        facts.append(f"thermal_fault_count={data.thermal_fault_count}")
    if data.position_mm is not None:
        position = data.position_mm
        facts.append(f"position_mm=({position.x:.3f},{position.y:.3f},{position.z:.3f})")
    return tuple(facts)


def _endstop_observations(data: EndstopData) -> tuple[str, ...]:
    facts = [
        f"{row.axis}: triggered={str(row.triggered).lower()}, "
        f"geometric_triggered={str(row.geometric_triggered).lower()}, enabled=true"
        for row in data.components
        if row.enabled is True and row.triggered is not None and row.geometric_triggered is not None
    ]
    return tuple(facts)


def project_receipts(results: Sequence[ToolResult]) -> tuple[ToolExecutionEvidence, ...]:
    """Project only this diagnosis's trusted live device reads into receipts.

    Failed reads, mock projections, knowledge fragments and empty projections
    never become receipts; ``verify_evidence`` re-checks the type and source.
    """
    receipts: list[ToolExecutionEvidence] = []
    for result in results:
        if result.source != "live_read" or result.status == "error" or result.evidence_id is None:
            continue
        if result.tool not in _DEVICE_TOOLS:
            continue
        if isinstance(result.data, PrinterStateProjection):
            observations = _state_observations(result.data)
        elif isinstance(result.data, EndstopData):
            observations = _endstop_observations(result.data)
        else:
            continue
        if observations:
            receipts.append(ToolExecutionEvidence(result.evidence_id, "live_read", result.status, observations))
    return tuple(receipts)


def _knowledge_keys(results: Sequence[ToolResult]) -> list[EvidenceKey]:
    keys: list[EvidenceKey] = []
    for result in results:
        if result.source != "curated_knowledge" or result.status == "error" or result.evidence_id is None:
            continue
        if isinstance(result.data, KnowledgeData):
            keys.extend(
                EvidenceKey(evidence_id=result.evidence_id, hit_index=index)
                for index in range(len(result.data.hits))
            )
    return keys


def _clip(text: str, limit: int = _MAX_TEXT) -> str:
    stripped = text.strip()
    return stripped[:limit].rstrip()


def _with_notes(report: DiagnosisReport, notes: Sequence[str]) -> DiagnosisReport:
    existing = {item.detail for item in report.limitations}
    additions: list[Limitation] = []
    for note in notes:
        detail = _clip(note)
        if not detail or detail in existing or len(report.limitations) + len(additions) + 1 > 25:
            continue
        additions.append(Limitation(detail=detail))
        existing.add(detail)
    if not additions:
        return report
    return report.model_copy(update={"limitations": [*report.limitations, *additions]})


def _degraded(diagnosis_id: str, *, results: Sequence[ToolResult], mock: bool, failure: ErrorCode,
              note: str) -> DiagnosisReport:
    report = build_report(diagnosis_id, mock=mock, results=list(results), draft=None, failure=failure)
    return _with_notes(report, (note,))


def adapt_sft_report(
    diagnosis_id: str,
    content: str,
    *,
    results: Sequence[ToolResult],
    user_observation: str,
    mock: bool,
    failure: ErrorCode | None = None,
    extra_limitations: Sequence[str] = (),
    v2_config=None,
    presentation: dict | None = None,
) -> DiagnosisReport:
    """Validate, normalize and verify one SFT draft, then build the API report.

    The draft's status is never upgraded: an explicit ``证据不足`` stays
    insufficient, and a model-claimed tool observation only becomes evidence
    when ``verify_evidence`` accepts an executor-issued receipt from this run.
    """
    results = list(results)
    if failure is not None:
        return _with_notes(_degraded(diagnosis_id, results=results, mock=mock, failure=failure,
                                     note="诊断流程未完成，LoRA 草稿未参与报告。"), extra_limitations)
    try:
        draft = validate_diagnosis(content)
    except DiagnosisValidationError as exc:
        return _with_notes(_degraded(
            diagnosis_id, results=results, mock=mock, failure="INVALID_PROVIDER_RESPONSE",
            note=f"LoRA 草稿未通过严格校验（{exc.code}），已降级为证据不足。",
        ), extra_limitations)

    safety = review_draft(draft)
    risky_fields = set(safety.fields)
    label = "类别待人工复核" if "fault_category" in risky_fields else draft.fault_category
    category = normalize_fault_category(label)
    receipts = project_receipts(results)
    verified = verify_evidence(draft, user_observations=(user_observation,), tool_results=receipts)

    notes: list[str] = []
    if safety.triggered:
        notes.append(safety.summary())
    if category.canonical is None and "fault_category" not in risky_fields:
        notes.append(f"模型给出的故障类别「{_clip(category.raw, 120)}」未映射到冻结词表，仅按原文展示。")
    elif category.match == "alias":
        notes.append("模型类别经冻结词表别名归一到规范标签。")
    if draft.tool_observation_available and not verified.tool_observation_available:
        notes.append("模型草稿声称有工具观测，但本轮没有执行器签发的可信设备回执，该声明不计为证据。")
    if "supporting_observation" not in risky_fields:
        notes.append(f"模型草稿声称的观察（未经本轮设备验证）：{_clip(draft.supporting_observation, 300)}")
    if "safety_note" not in risky_fields:
        notes.append(f"模型草稿安全提示：{_clip(draft.safety_note, 300)}")
    safe_recommendations = list(safe_checks(draft, safety))
    if safe_recommendations:
        notes.append(
            "模型草稿建议（未经工具验证）："
            + "；".join(_clip(check, 160) for check in safe_recommendations[:2]))

    device_keys = [
        EvidenceKey(evidence_id=receipt.evidence_id)
        for receipt in verified.device_observations
        if receipt.evidence_id is not None
    ]
    knowledge_keys = _knowledge_keys(results)
    cause_is_safe = "suspected_cause" not in risky_fields
    inference_text = (_clip(draft.suspected_cause, 400) if cause_is_safe
                      else "模型推测因安全审查未展开（见限制说明）。")
    hypotheses: list[Hypothesis] = []
    if verified.diagnosis_status == "待验证" and device_keys and knowledge_keys:
        hypotheses.append(Hypothesis(
            statement=f"{_clip(category.canonical or category.raw, 200)}（待核查）",
            inference=inference_text,
            evidence_refs=(device_keys + knowledge_keys)[:6],
        ))

    summary_cause = (_clip(draft.suspected_cause, 380) if cause_is_safe
                     else "模型推测因安全审查未展开")
    draft_report = ReportDraft(
        summary=f"{_clip(label, 80)}：{summary_cause}"[:_MAX_SUMMARY],
        hypotheses=hypotheses,
        evidence_refs=(device_keys + knowledge_keys)[:30],
        limitations=[_clip(note) for note in notes][:8],
        recommended_checks=["review_observations"] if device_keys else ["request_details"],
    )
    report = build_report(diagnosis_id, mock=mock, results=results, draft=draft_report)
    # build_report only forwards draft limitations when a hypothesis is supported;
    # a degraded/insufficient LoRA report must still carry the model-claim notes.
    report = _with_notes(report, notes)
    report = _with_notes(report, extra_limitations)
    if report.status == "complete":
        report = report.model_copy(update={"status": "partial"})
        report = _with_notes(report, ("LoRA 草稿不提供可核对的证据引用，结论保持部分支持。",))
    # V2 is opt-in only; a caller may pass a per-request config. Without one the
    # legacy environment switch is used, so V0.3 and historical runs are unchanged.
    from printdiag.reliability.v2.config import V2Config

    active_v2 = v2_config if v2_config is not None else V2Config.from_environment()
    assessment = None
    if active_v2.enabled:
        from printdiag.reliability.v2.evidence_policy import apply_v2_to_report

        rag_hits = [hit for result in results if isinstance(result.data, KnowledgeData)
                    for hit in result.data.hits]
        report, assessment = apply_v2_to_report(
            report, draft=draft, symptom=user_observation, results=results,
            rag_hits=rag_hits, config=active_v2)
    if presentation is not None and not safety.triggered:
        # This is a display hint only. Use user wording for the B01 symptom
        # pattern and never promote a model-claimed observation to a tool receipt.
        symptom = user_observation
        output_pattern = (("欠挤出" in symptom or "挤出不足" in symptom) or (
            any(term in symptom for term in ("外壁", "走线"))
            and "偏细" in symptom and any(term in symptom for term in ("空隙", "缝隙"))))
        preliminary = assessment is not None and (
            assessment.decision == "supported" or (
                assessment.decision == "hypothesis_only"
                and category.canonical == "挤出不足" and output_pattern))
        presentation.update(category=category.canonical if preliminary else None,
                            checks=safe_recommendations if preliminary else [],
                            safety_note=draft.safety_note if preliminary else None)
    return report
