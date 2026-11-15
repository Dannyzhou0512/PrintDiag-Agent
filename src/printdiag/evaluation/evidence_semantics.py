"""P10-5D evidence semantics and claim-support checks (candidate V2, not wired in).

The frozen P10-5C baseline intentionally does not use this module; it is a new
reliability version to be compared against the frozen baseline before adoption.
Core point: ``Endstop=false`` means "switch not triggered", not "the printer is
broken"; a hypothesis is only supported when the observed values actually show
an anomaly consistent with the claimed category.
"""

from dataclasses import dataclass

from printdiag.schemas.agent import (
    DiagnosisReport, EndstopData, PrinterStateProjection, ToolResult,
)


@dataclass(frozen=True)
class EvidenceSemantics:
    endstops: dict[str, dict[str, bool | None]]
    endstop_all_false: bool | None
    endstop_mismatch: bool | None
    idle: bool | None
    moves_queued: int | None
    thermal_fault_count: int | None
    hotend_temperature: float | None
    hotend_ambient_temperature: float | None
    thermal_normal_like: bool | None
    has_device_receipts: bool


def summarize_evidence(results: list[ToolResult]) -> EvidenceSemantics:
    endstops: dict[str, dict[str, bool | None]] = {}
    idle = moves = fault_count = hotend = ambient = None
    for result in results:
        data = result.data
        if isinstance(data, EndstopData):
            for component in data.components:
                endstops[component.axis] = {
                    "triggered": component.triggered if component.triggered is not None else None,
                    "geometric_triggered": (component.geometric_triggered
                                            if component.geometric_triggered is not None else None),
                }
        elif isinstance(data, PrinterStateProjection):
            idle = data.idle
            moves = data.moves_queued
            fault_count = data.thermal_fault_count
            hotend = data.hotend_temperature
            ambient = data.hotend_ambient_temperature
    known = [value for values in endstops.values()
             for value in (values["triggered"], values["geometric_triggered"])
             if value is not None]
    all_false = (all(value is False for value in known)
                 if endstops and len(known) == 2 * len(endstops) else None)
    mismatch = None
    if endstops:
        pairs = [(values["triggered"], values["geometric_triggered"]) for values in endstops.values()]
        if all(left is not None and right is not None for left, right in pairs):
            mismatch = any(left != right for left, right in pairs)
    thermal_normal_like = None
    if fault_count is not None and hotend is not None and ambient is not None:
        thermal_normal_like = fault_count == 0 and hotend <= ambient + 5.0
    elif fault_count is not None and fault_count > 0:
        thermal_normal_like = False
    return EvidenceSemantics(
        endstops=endstops, endstop_all_false=all_false, endstop_mismatch=mismatch,
        idle=idle, moves_queued=moves, thermal_fault_count=fault_count,
        hotend_temperature=hotend, hotend_ambient_temperature=ambient,
        thermal_normal_like=thermal_normal_like,
        has_device_receipts=bool(known) or idle is not None or fault_count is not None
        or hotend is not None,
    )


@dataclass(frozen=True)
class ClaimFinding:
    code: str
    statement: str
    reason: str


def assess_claim_support(report: DiagnosisReport, results: list[ToolResult]) -> tuple[ClaimFinding, ...]:
    semantics = summarize_evidence(results)
    findings: list[ClaimFinding] = []
    for hypothesis in report.hypotheses:
        statement = hypothesis.statement
        if any(term in statement for term in ("限位", "endstop", "Endstop")):
            if semantics.endstops and semantics.endstop_mismatch is False:
                reason = ("all endstop values are false and agree; false does not imply a "
                          "limit-switch fault" if semantics.endstop_all_false
                          else "all endstop triggered/geometric values agree; "
                               "no mismatch is observed")
                findings.append(ClaimFinding("UNSUPPORTED_ENDPOINT_CLAIM", statement, reason))
        if any(term in statement for term in ("运动", "电机", "丢步", "motion")):
            if semantics.idle is True and semantics.moves_queued == 0:
                findings.append(ClaimFinding(
                    "UNSUPPORTED_MOTION_CLAIM", statement,
                    "printer is idle with an empty queue; this does not evidence a motion fault"))
        if any(term in statement for term in ("温度", "热", "加热")):
            if semantics.thermal_normal_like is True:
                findings.append(ClaimFinding(
                    "UNSUPPORTED_THERMAL_CLAIM", statement,
                    "thermal fault count is zero and hotend temperature is near ambient"))
    return tuple(findings)


def apply_reliability_guard(report: DiagnosisReport, results: list[ToolResult]) -> DiagnosisReport:
    """Conservative candidate behaviour: unsupported claims downgrade the report."""
    findings = assess_claim_support(report, results)
    if not findings:
        return report
    from printdiag.schemas.agent import Limitation

    detail = "RELIABILITY_V2：模型假设缺乏证据支持（" + ", ".join(
        sorted({finding.code for finding in findings})) + "），已保守降级。"
    return report.model_copy(update={
        "status": "insufficient_evidence",
        "limitations": [*report.limitations, Limitation(detail=detail)],
    })
