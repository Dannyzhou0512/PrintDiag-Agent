"""Keep model claims separate from caller and tool-runner observations."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from printdiag.schemas.sft_diagnosis import DiagnosisStatus, SFTDiagnosisDraft

ToolSource = Literal["live_read", "mock", "curated_knowledge"]
ToolStatus = Literal["ok", "partial", "error"]


def _validate_observations(values: tuple[str, ...]) -> None:
    if type(values) is not tuple or any(type(item) is not str or not item.strip() for item in values):
        raise ValueError("observations must be a tuple of nonempty strings")


@dataclass(frozen=True, slots=True)
class ToolExecutionEvidence:
    """An adapter-owned receipt from the current diagnosis, never decoded from model JSON."""

    evidence_id: str | None
    source: ToolSource
    status: ToolStatus
    observations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source not in ("live_read", "mock", "curated_knowledge") or self.status not in ("ok", "partial", "error"):
            raise ValueError("invalid tool source or status")
        if self.evidence_id is not None and (type(self.evidence_id) is not str or not self.evidence_id.strip()):
            raise ValueError("invalid tool evidence ID")
        _validate_observations(self.observations)


@dataclass(frozen=True, slots=True)
class VerifiedEvidence:
    user_observations: tuple[str, ...]
    device_observations: tuple[ToolExecutionEvidence, ...]
    model_claimed_observation: str
    model_hypothesis: str
    model_claimed_tool_observation_available: bool
    diagnosis_status: DiagnosisStatus

    @property
    def tool_observation_available(self) -> bool:
        return bool(self.device_observations)


def verify_evidence(
    draft: SFTDiagnosisDraft,
    *,
    user_observations: tuple[str, ...] = (),
    tool_results: Iterable[ToolExecutionEvidence] = (),
) -> VerifiedEvidence:
    """Accept only current, projected receipts supplied by the trusted caller."""
    _validate_observations(user_observations)
    verified = []
    for result in tool_results:
        if type(result) is not ToolExecutionEvidence:
            raise TypeError("tool results must be executor-owned receipts")
        if (result.source == "live_read" and result.status in ("ok", "partial")
                and result.evidence_id is not None and result.observations):
            verified.append(result)
    status = draft.diagnosis_status if user_observations or verified else "证据不足"
    return VerifiedEvidence(
        user_observations=user_observations, device_observations=tuple(verified),
        model_claimed_observation=draft.supporting_observation, model_hypothesis=draft.suspected_cause,
        model_claimed_tool_observation_available=draft.tool_observation_available, diagnosis_status=status,
    )
