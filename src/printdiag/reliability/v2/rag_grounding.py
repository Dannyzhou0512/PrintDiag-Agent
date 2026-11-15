"""V2 RAG evidence grounding.

RAG hits are background/check guidance only. They are never device evidence and
never upgrade a diagnosis. This module records conflicts between retrieved
knowledge and observed device semantics.
"""

from dataclasses import dataclass
from typing import Sequence

from printdiag.evaluation.evidence_semantics import summarize_evidence
from printdiag.schemas.agent import ToolResult
from printdiag.schemas.sft_diagnosis import SFTDiagnosisDraft


@dataclass(frozen=True)
class RagGroundingResult:
    codes: tuple[str, ...]
    reasons: tuple[str, ...]
    rag_hits: int
    has_device_receipts: bool

    def as_dict(self) -> dict:
        return {"codes": list(self.codes), "reasons": list(self.reasons),
                "rag_hits": self.rag_hits,
                "has_device_receipts": self.has_device_receipts}


def ground_rag_claims(draft: SFTDiagnosisDraft, *, results: Sequence[ToolResult] = (),
                      rag_hits: Sequence = ()) -> RagGroundingResult:
    semantics = summarize_evidence(list(results))
    codes: list[str] = []
    reasons: list[str] = []
    if rag_hits:
        codes.append("RAG_BACKGROUND_ONLY")
        reasons.append("retrieved maintenance knowledge is background only, not a device receipt")
        if semantics.has_device_receipts and semantics.endstops:
            claim = f"{draft.fault_category} {draft.suspected_cause}"
            if any(term in claim for term in ("限位", "endstop")) and (
                    semantics.endstop_mismatch is False):
                codes.append("RAG_CONFLICTS_WITH_DEVICE")
                reasons.append("RAG knowledge mentions endstop checks, but device receipts show "
                               "agreeing/false endstops and no anomaly")
        if semantics.has_device_receipts and semantics.thermal_normal_like is True:
            claim = f"{draft.fault_category} {draft.suspected_cause}"
            if any(term in claim for term in ("温度", "加热", "热敏")):
                codes.append("RAG_CONFLICTS_WITH_DEVICE")
                reasons.append("RAG thermal knowledge is present, but observed hotend is near "
                               "ambient with zero thermal faults")
    return RagGroundingResult(tuple(codes), tuple(reasons), len(rag_hits),
                              semantics.has_device_receipts)
