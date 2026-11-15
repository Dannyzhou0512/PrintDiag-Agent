"""V2 package exports (opt-in; default V0.3 behaviour is unchanged)."""

from printdiag.reliability.v2.config import V2Config
from printdiag.reliability.v2.evidence_policy import (
    DraftEvidenceAssessment, apply_v2_to_report, assess_draft_evidence,
)
from printdiag.reliability.v2.output_guard import GuardResult, guard_output
from printdiag.reliability.v2.taxonomy import audit_taxonomy

__all__ = [
    "V2Config", "DraftEvidenceAssessment", "assess_draft_evidence", "apply_v2_to_report",
    "GuardResult", "guard_output", "audit_taxonomy",
]
