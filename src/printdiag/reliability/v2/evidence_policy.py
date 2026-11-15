"""V2 evidence-sufficiency policy.

Separates "no tool receipt" from "unsupported diagnosis". A clear user symptom
without a tool receipt may still justify a 待验证 hypothesis; a normal device
reading contradictory to the claimed anomaly must not be presented as supported.
The policy never edits model output and never uses dataset labels.
"""

from dataclasses import dataclass
import re
from typing import Iterable, Literal, Sequence

from printdiag.evaluation.evidence_semantics import summarize_evidence
from printdiag.reliability.v2.config import V2Config
from printdiag.schemas.agent import DiagnosisReport, ToolResult
from printdiag.schemas.sft_diagnosis import SFTDiagnosisDraft

Decision = Literal["supported", "hypothesis_only", "insufficient", "contradicted"]

_ANOMALY_TERMS = ("偏细", "拉丝", "堵塞", "堵头", "不粘", "翘边", "错位", "过热", "温度异常",
                  "挤出", "断裂", "缺失", "粗糙", "振纹", "疙瘩", "缝隙", "缺口", "划痕",
                  "鼓包", "孔洞", "倾斜", "翘曲", "脱离", "越界", "停止", "不出丝", "开裂",
                  "丢步", "异响", "抖动", "卡滞")
_COMMAND_ABSENCE = ("未收到", "没有收到", "并未收到", "未下发", "没有下发", "未发送", "没有发送")
_COMMAND_WORDS = ("指令", "命令")
_UNVERIFIED = ("没有确认", "未确认", "不确定", "不知道", "未核实", "没有核实")
_UNVERIFIED_OBJECTS = ("切片", "文件", "参数", "相同", "是否", "耗材")
_INSUFFICIENT_CONTEXT = (
    "未提供", "没有提供", "没有错误码", "没有日志", "没有温度或步进状态",
    "未提供升温曲线", "刚发送预热命令", "尚未开始", "还没有开始", "没有任何出丝",
    "没有出丝或送料记录", "待机",
)
_GENERIC_CONTEXT = frozenset(("只知道这些信息", "只有这些信息", "this is all i know", "that is all i know"))
_BARE_STATUS_REQUEST = re.compile(
    r"(?:请)?(?:诊断|判断|检查|查看|确认)(?:一下)?(?:打印机|设备)?(?:的)?(?:当前)?状态"
    r"|(?:please )?(?:check|assess|diagnose) (?:the )?(?:printer|device) status"
)


@dataclass(frozen=True)
class DraftEvidenceAssessment:
    decision: Decision
    codes: tuple[str, ...]
    reasons: tuple[str, ...]
    has_device_receipts: bool
    rag_only: bool

    @property
    def downgrade(self) -> bool:
        return self.decision in ("insufficient", "contradicted")


def _contains(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _only_generic_context(symptom: str) -> bool:
    # A qualifier adds no evidence of its own. Treat it as insufficient only
    # when the input contains no other text beyond a bare status request.
    clauses = [" ".join(clause.casefold().split())
               for clause in re.split(r"[，,。.!！?？；;\n]+", symptom) if clause.strip()]
    return any(clause in _GENERIC_CONTEXT for clause in clauses) and all(
        clause in _GENERIC_CONTEXT or _BARE_STATUS_REQUEST.fullmatch(clause) for clause in clauses
    )


def assess_draft_evidence(draft: SFTDiagnosisDraft, *, symptom: str,
                          results: Sequence[ToolResult] = (), rag_hits: Sequence = (),
                          config: V2Config | None = None) -> DraftEvidenceAssessment:
    cfg = config or V2Config()
    # ToolResult validates executor provenance for live reads. Mock values
    # cannot support or contradict a claim about the current device.
    semantics = summarize_evidence([result for result in results
                                    if result.source == "live_read" and result.status != "error"])
    claim_text = f"{draft.fault_category} {draft.suspected_cause}"
    motion_claim = _contains(claim_text, ("运动", "电机", "X轴", "Y轴", "Z轴", "丢步", "卡滞", "驱动"))
    endpoint_claim = _contains(claim_text, ("限位", "endstop", "Endstop"))
    thermal_claim = _contains(claim_text, ("温度", "加热", "热敏", "过热"))
    explicit_symptom = _contains(f"{symptom} {draft.supporting_observation}", _ANOMALY_TERMS)
    command_absence = _contains(symptom, _COMMAND_ABSENCE) and _contains(symptom, _COMMAND_WORDS)
    unverified = (_contains(symptom, _UNVERIFIED)
                  and _contains(symptom, _UNVERIFIED_OBJECTS))
    rag_only = bool(rag_hits) and not semantics.has_device_receipts
    codes: list[str] = []
    reasons: list[str] = []

    if draft.fault_category == "证据不足或正常状态":
        codes.append("EXPLICIT_INSUFFICIENT")
        reasons.append("model already selected the insufficient/normal category")
        return DraftEvidenceAssessment("insufficient", tuple(codes), tuple(reasons),
                                       semantics.has_device_receipts, rag_only)
    if motion_claim and command_absence:
        codes.extend(("COMMAND_ABSENCE", "MOTION_CLAIM_WITHOUT_COMMAND"))
        reasons.append("no motion command was sent, so missing motion does not evidence a "
                       "motion-control fault")
        return DraftEvidenceAssessment("insufficient", tuple(codes), tuple(reasons),
                                       semantics.has_device_receipts, rag_only)
    if unverified:
        codes.append("UNVERIFIED_CONTEXT")
        reasons.append("the symptom explicitly says a key condition was not verified; a fault "
                       "hypothesis must stay evidence-insufficient until it is checked")
        return DraftEvidenceAssessment("insufficient", tuple(codes), tuple(reasons),
                                       semantics.has_device_receipts, rag_only)
    if _contains(symptom, _INSUFFICIENT_CONTEXT):
        codes.append("INSUFFICIENT_CONTEXT")
        reasons.append("the symptom states that key evidence (logs, photos, parameters, warm-up "
                       "time, or measurements) is missing")
        return DraftEvidenceAssessment("insufficient", tuple(codes), tuple(reasons),
                                       semantics.has_device_receipts, rag_only)
    if not semantics.has_device_receipts and _only_generic_context(symptom):
        codes.append("INSUFFICIENT_CONTEXT")
        reasons.append("the input contains only a generic qualifier or status request without "
                       "an observed symptom or trusted device receipt")
        return DraftEvidenceAssessment("insufficient", tuple(codes), tuple(reasons), False, rag_only)
    if rag_only:
        codes.append("RAG_BACKGROUND_ONLY")
        reasons.append("RAG knowledge is background only and is not a device receipt")
    if endpoint_claim:
        if semantics.endstop_mismatch is True:
            codes.append("ENDSTOP_MISMATCH_OBSERVED")
            reasons.append("triggered vs geometric endstop values actually differ")
            return DraftEvidenceAssessment("supported", tuple(codes), tuple(reasons),
                                           semantics.has_device_receipts, rag_only)
        if semantics.endstops and semantics.endstop_mismatch is False:
            codes.append("NORMAL_ENDPOINT_CONTRADICTS_CLAIM")
            reasons.append("all endstop values agree (typically false); no limit-switch anomaly "
                           "is supported")
            return DraftEvidenceAssessment("contradicted", tuple(codes), tuple(reasons),
                                           semantics.has_device_receipts, rag_only)
        if not semantics.has_device_receipts:
            codes.append("NO_TOOL_RECEIPT")
            reasons.append("no trusted executor receipt; the endpoint hypothesis is unverified")
            return DraftEvidenceAssessment("hypothesis_only", tuple(codes), tuple(reasons),
                                           False, rag_only)
    if thermal_claim:
        if semantics.thermal_normal_like is True:
            codes.append("NORMAL_THERMAL_CONTRADICTS_CLAIM")
            reasons.append("thermal fault count is zero and the hotend is near ambient")
            return DraftEvidenceAssessment("contradicted", tuple(codes), tuple(reasons),
                                           semantics.has_device_receipts, rag_only)
        if semantics.thermal_fault_count is not None and semantics.thermal_fault_count > 0:
            codes.append("THERMAL_FAULT_OBSERVED")
            return DraftEvidenceAssessment("supported", tuple(codes), tuple(reasons),
                                           semantics.has_device_receipts, rag_only)
        if not semantics.has_device_receipts:
            codes.append("NO_TOOL_RECEIPT")
            return DraftEvidenceAssessment("hypothesis_only", tuple(codes), tuple(reasons),
                                           False, rag_only)
    if motion_claim and semantics.idle is True and semantics.moves_queued == 0:
        codes.append("IDLE_DOES_NOT_EVIDENCE_MOTION_FAULT")
        reasons.append("printer is idle with an empty queue; idle does not confirm a motion fault")
        return DraftEvidenceAssessment("hypothesis_only", tuple(codes), tuple(reasons),
                                       semantics.has_device_receipts, rag_only)
    if explicit_symptom and not semantics.has_device_receipts:
        codes.append("NO_TOOL_RECEIPT")
        reasons.append("clear user symptom but no trusted executor receipt; keep it as a "
                       "待验证 hypothesis, not a confirmed fault")
        return DraftEvidenceAssessment("hypothesis_only", tuple(codes), tuple(reasons),
                                       False, rag_only)
    if explicit_symptom and semantics.has_device_receipts:
        codes.append("DEVICE_BOUNDARY")
        reasons.append("clear user symptom, but current device receipts do not measure or "
                       "confirm this failure mode; keep it as a 待验证 hypothesis")
        return DraftEvidenceAssessment("hypothesis_only", tuple(codes), tuple(reasons),
                                       True, rag_only)
    normal_question = _contains(symptom, ("是否正常", "正常吗", "有没有问题", "是否连通"))
    if normal_question and not explicit_symptom and not semantics.has_device_receipts:
        codes.append("NO_ANOMALY_IN_SYMPTOM")
        reasons.append("the symptom only asks whether the device is normal and provides no "
                       "anomaly description; a fault hypothesis is unsupported")
        return DraftEvidenceAssessment("insufficient", tuple(codes), tuple(reasons),
                                       False, rag_only)
    if not semantics.has_device_receipts:
        codes.append("NO_TOOL_RECEIPT")
        reasons.append("no trusted executor receipt; keep the model's hypothesis as 待验证 "
                       "without upgrading it to confirmed evidence")
        return DraftEvidenceAssessment("hypothesis_only", tuple(codes), tuple(reasons),
                                       False, rag_only)
    if semantics.has_device_receipts:
        codes.append("NO_MATCHING_ANOMALY")
        reasons.append("device receipts do not show an anomaly matching the claim")
        return DraftEvidenceAssessment("contradicted", tuple(codes), tuple(reasons),
                                       True, rag_only)
    codes.append("INSUFFICIENT_EVIDENCE")
    reasons.append("neither a clear symptom nor a trusted device receipt supports a diagnosis")
    return DraftEvidenceAssessment("insufficient", tuple(codes), tuple(reasons), False, rag_only)


def apply_v2_to_report(report: DiagnosisReport, *, draft: SFTDiagnosisDraft, symptom: str,
                       results: Sequence[ToolResult] = (), rag_hits: Sequence = (),
                       config: V2Config | None = None) -> tuple[DiagnosisReport, DraftEvidenceAssessment]:
    """Return a V2-annotated copy; never mutates the frozen V0.3 report in place."""
    cfg = config or V2Config.from_environment()
    assessment = assess_draft_evidence(draft, symptom=symptom, results=results,
                                      rag_hits=rag_hits, config=cfg)
    if not cfg.enabled:
        return report, assessment
    from printdiag.schemas.agent import Limitation

    details = ["V2 证据充分性：" + "；".join(assessment.reasons)] if assessment.reasons else []
    details += [f"V2 编码：{code}" for code in assessment.codes]
    limitations = [*report.limitations, *(Limitation(detail=text[:480]) for text in details)]
    status = report.status
    if cfg.downgrade_contradicted and assessment.downgrade:
        status = "insufficient_evidence"
    return report.model_copy(update={"status": status, "limitations": limitations[:25]}), assessment
