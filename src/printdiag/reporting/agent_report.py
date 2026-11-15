"""Resolve model-selected evidence to observations issued by this diagnosis."""

import re
from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from printdiag.reporting.safety_review import SafetyReview, review_fields
from printdiag.schemas.agent import (
    Contract, DiagnosisReport, EndstopData, ErrorCode, EvidenceKey, EvidenceReference, Hypothesis,
    KnowledgeData, KnowledgeHit, Limitation, ShortText, ToolResult,
)

Check = Literal["inspect_mounting", "inspect_cable_power_off", "review_observations", "request_details"]
_CHECKS: dict[Check, str] = {
    "inspect_mounting": "设备静止后，目视检查限位开关安装及触发机构是否有松动或遮挡。",
    "inspect_cable_power_off": "设备断电后，按设备手册目视检查限位接头与线束是否松脱。",
    "review_observations": "人工核对设备状态的采样时间、队列和相关报错。",
    "request_details": "补充设备型号、故障出现条件和已有的报错信息。",
}
_ISSUES: dict[ErrorCode, str] = {
    "TOOL_NOT_ALLOWED": "拒绝调用未注册的工具。",
    "INVALID_ARGUMENT": "工具参数未通过校验。",
    "ACCESS_DENIED": "设备请求或重定向不符合只读访问策略。",
    "DEVICE_UNAVAILABLE": "设备通信不可用。",
    "DEVICE_TIMEOUT": "设备读取超时。",
    "DEVICE_HTTP_ERROR": "设备返回 HTTP 错误。",
    "INVALID_DEVICE_RESPONSE": "设备返回数据不完整或不符合协议。",
    "FIELD_UNAVAILABLE": "部分设备字段不可用。",
    "STALE_OBSERVATION": "超过采样窗口的设备字段已剔除。",
    "KNOWLEDGE_UNAVAILABLE": "维护知识索引不可用。",
    "PROVIDER_UNAVAILABLE": "诊断 Provider 不可用或请求超时。",
    "INVALID_PROVIDER_RESPONSE": "Provider 响应或证据引用未通过校验。",
    "BUDGET_EXHAUSTED": "诊断达到时间或调用上限，或检测到重复调用。",
}
_CONFIRMED_DAMAGE = re.compile(r"(?:已经确认|已确认|确定是).{0,12}(?:损坏|坏了)|confirmed hardware damage", re.I)


class ReportDraft(Contract):
    summary: ShortText
    hypotheses: Annotated[list[Hypothesis], Field(max_length=4)]
    evidence_refs: Annotated[list[EvidenceKey], Field(max_length=30)]
    limitations: Annotated[list[ShortText], Field(max_length=8)]
    recommended_checks: Annotated[list[Check], Field(max_length=4)]

    @model_validator(mode="after")
    def reject_unsupported_damage_claim(self) -> Self:
        text = [self.summary, *self.limitations]
        text.extend(value for item in self.hypotheses for value in (item.statement, item.inference))
        if any(_CONFIRMED_DAMAGE.search(value) for value in text):
            raise ValueError("physical damage is outside the evidence contract")
        return self


def _review_draft(draft: ReportDraft) -> tuple[ReportDraft, SafetyReview]:
    fields = [("summary", draft.summary)]
    fields.extend((f"limitation.{i}", text) for i, text in enumerate(draft.limitations))
    fields.extend((f"hypothesis.{i}.{field}", getattr(hypothesis, field))
                  for i, hypothesis in enumerate(draft.hypotheses) for field in ("statement", "inference"))
    safety = review_fields(fields)
    if not safety.triggered:
        return draft, safety
    risky = set(safety.fields)
    withheld = "模型文字已隐藏，需人工复核。"
    hypotheses = [
        hypothesis.model_copy(update={field: withheld for field in ("statement", "inference")
                                      if f"hypothesis.{i}.{field}" in risky})
        for i, hypothesis in enumerate(draft.hypotheses)
    ]
    return draft.model_copy(update={
        "summary": withheld if "summary" in risky else draft.summary,
        "hypotheses": hypotheses,
        "limitations": [text for i, text in enumerate(draft.limitations) if f"limitation.{i}" not in risky],
    }), safety


def _catalog(results: list[ToolResult]) -> dict[tuple[str, int | None], EvidenceReference]:
    catalog = {}
    for result in results:
        if result.evidence_id is None or result.observed_at_utc is None or result.data is None:
            continue
        items = enumerate(result.data.hits) if isinstance(result.data, KnowledgeData) else [(None, result.data)]
        for index, data in items:
            reference = EvidenceReference(
                evidence_id=result.evidence_id, hit_index=index, tool=result.tool, source=result.source,
                observed_at_utc=result.observed_at_utc, data=data,
            )
            catalog[reference.key()] = reference
    return catalog


def _has_device_observation(reference: EvidenceReference) -> bool:
    data = reference.data
    if isinstance(data, EndstopData):
        return any(row.enabled is True and row.triggered is not None and row.geometric_triggered is not None
                   for row in data.components)
    return not isinstance(data, KnowledgeHit)


def build_report(diagnosis_id: str, *, mock: bool, results: list[ToolResult],
                 draft: ReportDraft | None, failure: ErrorCode | None = None,
                 extra_limitations: Sequence[str] = ()) -> DiagnosisReport:
    catalog = _catalog(results)
    requested = []
    if draft is not None:
        requested = [ref.key() for ref in draft.evidence_refs]
        requested.extend(ref.key() for hypothesis in draft.hypotheses for ref in hypothesis.evidence_refs)
        if any(key not in catalog for key in requested):
            failure = "INVALID_PROVIDER_RESPONSE"
            draft = None

    codes = {issue.code for result in results for issue in result.issues}
    if failure is not None:
        codes.add(failure)
    limitations = []
    if draft is not None:
        draft, safety = _review_draft(draft)
        if safety.triggered:
            limitations.append(Limitation(detail=safety.summary()))
    limitations.extend(Limitation(code=code, detail=_ISSUES[code]) for code in sorted(codes))
    if mock:
        limitations.append(Limitation(detail="使用确定性的 Mock Provider，本次结果来自软件演练。"))
    if any(result.provenance == "simulator" for result in results):
        limitations.append(Limitation(detail="设备观测来自本地 Marlin Simulator（模拟器），不等同实体打印机测量。"))
    limitations.append(Limitation(detail="只读观测和知识片段仅支持待验证假设，无法确认实体硬件根因。"))
    for detail in extra_limitations:
        text = detail.strip()
        if text and len(limitations) < 25:
            limitations.append(Limitation(detail=text[:500]))
    if any(isinstance(result.data, KnowledgeData) and not result.data.hits for result in results):
        limitations.append(Limitation(detail="维护知识检索无匹配。"))

    supported = []
    full_evidence_ids = {result.evidence_id for result in results if result.status == "ok"}
    if draft is not None:
        for hypothesis in draft.hypotheses:
            refs = [catalog[ref.key()] for ref in hypothesis.evidence_refs]
            if any(_has_device_observation(ref) and (mock or ref.evidence_id in full_evidence_ids)
                   for ref in refs) and any(
                       ref.source == "curated_knowledge" for ref in refs
            ):
                supported.append(hypothesis)
    if supported:
        all_supported = len(supported) == len(draft.hypotheses)
        status = "partial" if codes or not all_supported else "complete"
        if all_supported:
            limitations.extend(Limitation(detail=detail) for detail in draft.limitations)
            summary = "初步诊断（待核查） " + draft.summary[:486].rstrip()
        else:
            summary = "仅保留能由本轮设备观测和维护知识共同支持的待核查假设。"
        checks = draft.recommended_checks
    else:
        status = "insufficient_evidence"
        summary = "当前设备观测和维护知识不足以支持可核查的故障假设。"
        checks = ["review_observations", "request_details"]
    refs = [catalog[key] for key in dict.fromkeys(requested)] if draft is not None else list(catalog.values())
    return DiagnosisReport(
        diagnosis_id=diagnosis_id, status=status, summary=summary, hypotheses=supported,
        evidence_refs=refs, limitations=limitations[:25],
        recommended_checks=[_CHECKS[check] for check in dict.fromkeys(checks)],
    )
