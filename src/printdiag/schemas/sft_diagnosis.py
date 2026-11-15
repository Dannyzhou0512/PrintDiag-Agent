"""Strict boundary for Qwen SFT drafts, separate from the Agent API report."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Literal
import unicodedata

DiagnosisStatus = Literal["待验证", "证据不足"]
_TEXT_FIELDS = ("fault_category", "suspected_cause", "supporting_observation", "safety_note")
DIAGNOSIS_FIELDS = frozenset((*_TEXT_FIELDS, "recommended_checks", "diagnosis_status", "tool_observation_available"))


class DiagnosisValidationError(ValueError):
    def __init__(self, code: str, field: str | None = None) -> None:
        self.code = code
        self.field = field
        super().__init__(f"{code} ({field})" if field else code)


@dataclass(frozen=True, slots=True)
class SFTDiagnosisDraft:
    fault_category: str
    suspected_cause: str
    supporting_observation: str
    recommended_checks: tuple[str, str]
    diagnosis_status: DiagnosisStatus
    tool_observation_available: bool
    safety_note: str


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise DiagnosisValidationError("duplicate_json_key")
    return result


def _reject_constant(_: str) -> None:
    raise DiagnosisValidationError("invalid_json")


def parse_diagnosis_json(text: str) -> dict[str, object]:
    """Parse one JSON object without ambiguous duplicate keys or NaN/Infinity."""
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except DiagnosisValidationError:
        raise
    except (RecursionError, ValueError):
        raise DiagnosisValidationError("invalid_json") from None
    if not isinstance(value, dict):
        raise DiagnosisValidationError("expected_object")
    return value


def _check_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return "".join(normalized.rstrip(".!?。！？,，;；、:：").split())


def validate_diagnosis(payload: str | Mapping[str, object]) -> SFTDiagnosisDraft:
    """Validate the seven model fields without coercion or automatic repair."""
    values = parse_diagnosis_json(payload) if isinstance(payload, str) else payload
    if not isinstance(values, Mapping):
        raise DiagnosisValidationError("expected_object")
    if values.keys() != DIAGNOSIS_FIELDS:
        raise DiagnosisValidationError("invalid_fields")
    for field in _TEXT_FIELDS:
        if type(values[field]) is not str or not values[field].strip():
            raise DiagnosisValidationError("invalid_text", field)
    if type(values["tool_observation_available"]) is not bool:
        raise DiagnosisValidationError("invalid_boolean", "tool_observation_available")
    if type(values["diagnosis_status"]) is not str or values["diagnosis_status"] not in ("待验证", "证据不足"):
        raise DiagnosisValidationError("invalid_status", "diagnosis_status")
    checks = values["recommended_checks"]
    if type(checks) is not list or len(checks) != 2 or any(type(item) is not str or not _check_key(item) for item in checks):
        raise DiagnosisValidationError("invalid_checks", "recommended_checks")
    if _check_key(checks[0]) == _check_key(checks[1]):
        raise DiagnosisValidationError("duplicate_checks", "recommended_checks")
    return SFTDiagnosisDraft(
        fault_category=values["fault_category"], suspected_cause=values["suspected_cause"],
        supporting_observation=values["supporting_observation"], recommended_checks=(checks[0], checks[1]),
        diagnosis_status=values["diagnosis_status"], tool_observation_available=values["tool_observation_available"],
        safety_note=values["safety_note"],
    )
