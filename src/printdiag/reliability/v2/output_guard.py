"""V2 structured-output guard.

No silent repair: invalid model output keeps its raw text and a controlled error.
A separate, explicitly-labelled repair candidate can be requested for checks-only
problems, but it is never counted as a strict pass.
"""

from dataclasses import dataclass
import json

from printdiag.schemas.sft_diagnosis import (
    DIAGNOSIS_FIELDS, DiagnosisValidationError, validate_diagnosis,
)


@dataclass(frozen=True)
class GuardResult:
    valid: bool
    raw_output: str
    error_code: str | None = None
    error_field: str | None = None
    repair_candidate: str | None = None
    repair_valid: bool = False
    repair_note: str | None = None

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "raw_output": self.raw_output,
            "error_code": self.error_code,
            "error_field": self.error_field,
            "repair_candidate": self.repair_candidate,
            "repair_valid": self.repair_valid,
            "repair_note": self.repair_note,
        }

    def as_public_dict(self) -> dict:
        """Expose validation diagnostics without publishing provider text."""
        return {**self.as_dict(), "raw_output": None, "repair_candidate": None}


def _checks_repair(raw: str) -> tuple[str | None, str | None]:
    """Return (candidate, note) for duplicate/too-many checks only."""
    try:
        obj = json.loads(raw)
    except ValueError:
        return None, "raw output is not valid JSON; no safe checks-only repair"
    if not isinstance(obj, dict) or set(obj) != DIAGNOSIS_FIELDS:
        return None, "field set is invalid; no checks-only repair"
    checks = obj.get("recommended_checks")
    if not isinstance(checks, list) or not all(isinstance(item, str) for item in checks):
        return None, "recommended_checks is not a string list"
    unique = []
    for item in checks:
        if item not in unique:
            unique.append(item)
    if len(unique) < 2:
        return None, "fewer than two distinct checks; cannot fabricate a second check"
    candidate = dict(obj)
    candidate["recommended_checks"] = unique[:2]
    return json.dumps(candidate, ensure_ascii=False), "checks de-duplicated and truncated to two"


def guard_output(raw: str, *, attempt_repair: bool = False) -> GuardResult:
    try:
        validate_diagnosis(raw)
        return GuardResult(valid=True, raw_output=raw)
    except DiagnosisValidationError as exc:
        repair = None
        repair_valid = False
        note = None
        if attempt_repair and exc.code in ("duplicate_checks", "invalid_checks"):
            repair, note = _checks_repair(raw)
            if repair is not None:
                try:
                    validate_diagnosis(repair)
                    repair_valid = True
                except DiagnosisValidationError:
                    repair_valid = False
        return GuardResult(
            valid=False, raw_output=raw, error_code=exc.code, error_field=exc.field,
            repair_candidate=repair, repair_valid=repair_valid, repair_note=note)
