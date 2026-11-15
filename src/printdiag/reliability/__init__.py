"""PrintDiag V2 reliability package.

All V2 behaviour is opt-in via ``PRINTDIAG_RELIABILITY_V2=1``; the frozen V0.3
baseline remains the default and is never modified by these modules.
"""

from printdiag.reliability.v2.config import V2Config
from printdiag.reliability.v2.evidence_policy import (
    DraftEvidenceAssessment, assess_draft_evidence,
)

__all__ = ["V2Config", "DraftEvidenceAssessment", "assess_draft_evidence"]
