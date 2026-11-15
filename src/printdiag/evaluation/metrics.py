"""Reusable, denominator-explicit metrics for P10-5D model comparison.

Every metric carries a numerator and denominator (or an explicit N/A reason);
nothing is fabricated when labels are missing. ``rows`` are per-case dicts with
at least: ``sample_id``, ``valid``, ``error``, ``category_canonical``,
``reference_category``, ``prediction_status``, ``reference_status``,
``latency_ms``. All rates use the sample set they were computed on.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping


def rate(numerator: int, denominator: int, *, definition: str,
         applicable_to: str) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": (numerator / denominator) if denominator else None,
        "definition": definition,
        "applicable_to": applicable_to,
        "na_reason": None if denominator else "denominator is zero",
    }


def na(reason: str, *, definition: str, applicable_to: str) -> dict:
    return {"numerator": None, "denominator": None, "value": None,
            "definition": definition, "applicable_to": applicable_to,
            "na_reason": reason}


def structural_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    valid = [row for row in rows if row.get("valid")]
    errors = Counter(row.get("error") for row in rows if not row.get("valid"))
    return {
        "samples": total,
        "json_syntax_valid": rate(
            sum(1 for row in rows if row.get("valid") or row.get("error")), total,
            definition="drafts that were parseable JSON objects regardless of schema",
            applicable_to="all generated drafts"),
        "strict_seven_field_pass": rate(
            len(valid), total,
            definition="drafts passing validate_diagnosis with exactly the 7 fields",
            applicable_to="all generated drafts"),
        "missing_field_or_extra_field": rate(
            errors.get("invalid_fields", 0), total,
            definition="drafts rejected for missing or extra fields",
            applicable_to="all generated drafts"),
        "invalid_boolean": rate(
            errors.get("invalid_boolean", 0), total,
            definition="drafts with non-JSON boolean tool_observation_available",
            applicable_to="all generated drafts"),
        "invalid_diagnosis_status": rate(
            errors.get("invalid_status", 0), total,
            definition="drafts whose diagnosis_status is not 待验证/证据不足",
            applicable_to="all generated drafts"),
        "duplicate_recommendation": rate(
            errors.get("duplicate_checks", 0), total,
            definition="drafts with duplicate recommended_checks",
            applicable_to="all generated drafts"),
        "unknown_category": rate(
            sum(1 for row in valid if row.get("category_canonical") is None), len(valid),
            definition="valid drafts whose category is outside the frozen taxonomy",
            applicable_to="valid drafts only"),
        "error_codes": dict(errors),
    }


def _f1_from_pairs(pairs: list[tuple[str, str]]) -> tuple[dict, float]:
    labels = sorted({reference for reference, _ in pairs} | {prediction for _, prediction in pairs})
    if not labels:
        return {}, 0.0
    per_label = {}
    for label in labels:
        tp = sum(1 for reference, prediction in pairs
                 if reference == label and prediction == label)
        fp = sum(1 for reference, prediction in pairs
                 if prediction == label and reference != label)
        fn = sum(1 for reference, prediction in pairs
                 if reference == label and prediction != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"support": tp + fn, "precision": round(precision, 4),
                            "recall": round(recall, 4), "f1": round(f1, 4)}
    return per_label, sum(item["f1"] for item in per_label.values()) / len(per_label)


def _f1_by_label(rows: list[dict]) -> tuple[dict, float]:
    return _f1_from_pairs([(row["reference_category"], row["category_canonical"])
                           for row in rows])


def category_metrics(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("valid")]
    # Full-sample accuracy: every sample with a reference label counts in the
    # denominator; invalid drafts and unmapped categories are wrong.
    full = [row for row in rows if row.get("reference_category") is not None]
    full_hits = sum(1 for row in full
                    if row.get("valid") and row.get("category_canonical") is not None
                    and row["category_canonical"] == row["reference_category"])
    mapped = [row for row in valid
              if row.get("reference_category") is not None
              and row.get("reference_in_taxonomy", True)
              and row.get("category_canonical") is not None]
    mapped_hits = sum(1 for row in mapped
                      if row["category_canonical"] == row["reference_category"])
    per_label, macro_f1 = _f1_by_label(mapped)
    # Full macro-F1 treats invalid/unmapped predictions as one explicit
    # INVALID_OR_UNMAPPED label, so they lower the score instead of disappearing.
    full_pairs = [(
        row["reference_category"],
        row["category_canonical"] if row.get("valid") and row.get("category_canonical")
        else "INVALID_OR_UNMAPPED",
    ) for row in full]
    _, macro_f1_full = _f1_from_pairs(full_pairs)
    confusion = Counter((row.get("reference_category"), row.get("category_canonical"))
                        for row in mapped)
    return {
        "fault_category_accuracy": rate(
            full_hits, len(full),
            definition="full-sample accuracy over every sample with a reference category; "
                       "invalid drafts and unmapped categories count as wrong",
            applicable_to="all samples with a reference category"),
        "mapped_subset_accuracy": rate(
            mapped_hits, len(mapped),
            definition="accuracy restricted to valid drafts whose category maps into the frozen "
                       "taxonomy; reported separately, never used to hide invalid/unmapped samples",
            applicable_to="valid drafts with both canonical and reference categories"),
        "mapping_coverage_rate": rate(
            len(mapped), len(full),
            definition="valid drafts with a canonical category / all reference-labelled samples",
            applicable_to="all samples with a reference category"),
        "macro_f1": {"value": round(macro_f1, 4) if mapped else None,
                     "denominator": len(mapped),
                     "definition": "unweighted mean of per-label F1 over union labels "
                                   "(mapped subset)",
                     "applicable_to": "valid drafts with a canonical category",
                     "na_reason": None if mapped else "no mapped valid drafts"},
        "macro_f1_full": {"value": round(macro_f1_full, 4) if full else None,
                          "denominator": len(full),
                          "definition": "macro-F1 over all reference-labelled samples; invalid/"
                                        "unmapped predictions form the INVALID_OR_UNMAPPED label",
                          "applicable_to": "all samples with a reference category",
                          "na_reason": None if full else "no reference-labelled samples"},
        "per_label": per_label,
        "confusion": {f"{ref} -> {pred}": count for (ref, pred), count in confusion.items()},
        "invalid_or_unmapped_samples": {
            "numerator": len(full) - full_hits, "denominator": len(full),
            "definition": "reference-labelled samples that are invalid or whose category is "
                          "unmapped/mismatched; all counted as wrong in full accuracy",
            "applicable_to": "all samples with a reference category"},
    }


def reliability_metrics(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("valid")]
    no_tool = [row for row in valid if row.get("tool_observation_available") is False]
    # Unsupported hypothesis requires a per-sample support judgement (semantic or
    # label). Absence of a tool receipt alone is NOT evidence of an unsupported
    # or hallucinated hypothesis.
    evaluable = [row for row in valid
                 if row.get("hypothesis_support") in ("supported", "unsupported")]
    unsupported = [row for row in evaluable if row["hypothesis_support"] == "unsupported"]
    insufficient_reference = [row for row in valid
                              if row.get("reference_status") == "证据不足"]
    insufficient_predicted = [row for row in insufficient_reference
                              if row.get("prediction_status") == "证据不足"]
    negatives = [row for row in valid
                 if row.get("reference_category") == "证据不足或正常状态"]
    hypothesis_negatives = [row for row in negatives
                            if row.get("category_canonical") != "证据不足或正常状态"]
    unsupported_hypothesis = (
        rate(len(unsupported), len(evaluable),
             definition="evaluable hypotheses judged unsupported by user symptoms and "
                        "trusted device evidence",
             applicable_to="valid drafts with hypothesis_support in supported/unsupported")
        if evaluable else na(
            "no sample carries a hypothesis-support judgement; text-only samples cannot "
            "determine whether a 待验证 hypothesis is unsupported",
            definition="evaluable hypotheses judged unsupported by user symptoms and "
                       "trusted device evidence",
            applicable_to="valid drafts with hypothesis_support in supported/unsupported"))
    return {
        "definition_note": "no_tool_observation_rate only means no executor receipt; "
                           "unsupported_hypothesis_rate is a separate, label-gated metric; "
                           "normal-state FPR is hypothesis-level, not confirmed-fault level",
        "no_tool_observation_rate": rate(
            len(no_tool), len(valid),
            definition="valid drafts whose tool_observation_available is false, i.e. no "
                       "trusted executor receipt for this run",
            applicable_to="valid drafts (all sets)"),
        "unsupported_hypothesis_rate": unsupported_hypothesis,
        "insufficient_evidence_recognition_rate": rate(
            len(insufficient_predicted), len(insufficient_reference),
            definition="reference diagnosis_status 证据不足 samples where the draft also "
                       "outputs 证据不足",
            applicable_to="valid drafts with reference status 证据不足"),
        "normal_state_hypothesis_false_positive_rate": rate(
            len(hypothesis_negatives), len(negatives),
            definition="normal/insufficient labelled samples where the draft raised a "
                       "non-insufficient fault-category hypothesis",
            applicable_to="valid drafts on samples labelled 证据不足或正常状态"),
        "confirmed_fault_claims": rate(
            0, len(valid),
            definition="drafts claiming a confirmed fault; the V0.3 contract has no confirmed "
                       "state, so only impossible claims would count",
            applicable_to="valid drafts"),
        "legacy_metric_migration": {
            "false_positive_rate": "renamed to normal_state_hypothesis_false_positive_rate; "
                                   "hypothesis-level, not device-level FPR",
            "unsupported_diagnosis_rate": "deprecated; split into no_tool_observation_rate "
                                          "and label-gated unsupported_hypothesis_rate",
            "insufficient_evidence_recognition": "renamed with _rate suffix",
        },
    }


def safety_version_metrics(rows: list[dict]) -> dict:
    """Compare two safety-review versions on manually labelled raw outputs.

    Each row needs ``old_blocked``, ``new_blocked`` (bool) and ``manual_danger``
    (bool or None). Without manual danger labels the danger/miss metrics are N/A;
    false-block counts still require labels, so all are N/A until reviewed.
    """
    labelled = [row for row in rows if row.get("manual_danger") is not None]
    dangers = [row for row in labelled if row["manual_danger"]]

    def blocks(version: str) -> dict:
        blocked = [row for row in labelled if row.get(f"{version}_blocked")]
        false_blocks = [row for row in blocked if not row["manual_danger"]]
        missed = [row for row in dangers if not row.get(f"{version}_blocked")]
        return {
            "blocked": len(blocked),
            "false_blocks": rate(
                len(false_blocks), len(labelled),
                definition=f"{version} safety review flagged a safe answer",
                applicable_to="manually reviewed pilot answers"),
            "missed_dangers": rate(
                len(missed), len(dangers),
                definition=f"answers manually judged dangerous that {version} did not flag",
                applicable_to="manually labelled dangerous answers"),
        }

    if not labelled:
        return {"samples": len(rows), "manual_danger_labels": 0,
                "na_reason": "no manual danger labels; cannot compute false blocks or misses"}
    return {
        "samples": len(rows),
        "manual_danger_labels": len(labelled),
        "dangerous_answers": len(dangers),
        "old": blocks("old"),
        "new": blocks("new"),
    }


def latency_metrics(rows: list[dict]) -> dict:
    values = sorted(row["latency_ms"] for row in rows
                    if isinstance(row.get("latency_ms"), (int, float)))
    if not values:
        return {"samples": 0, "na_reason": "no measured latency"}
    import statistics

    def at(quantile: float) -> float:
        index = min(len(values) - 1, max(0, int(round(quantile * len(values))) - 1))
        return round(values[index], 2)

    return {
        "samples": len(values),
        "mean_ms": round(statistics.fmean(values), 2),
        "p50_ms": round(statistics.median(values), 2),
        "p95_ms": at(0.95),
        "max_ms": round(values[-1], 2),
        "definition": "cloud inference latency reported by the service per request",
    }


def aggregate(rows: list[dict]) -> dict:
    return {
        "structural": structural_metrics(rows),
        "category": category_metrics(rows),
        "reliability": reliability_metrics(rows),
        "latency": latency_metrics(rows),
    }


def compare(reference: Iterable[Mapping], candidate: Iterable[Mapping]) -> dict:
    """Paired per-sample comparison used for LoRA/RAG net-gain analysis.

    Only pairs where both sides are valid and labelled contribute; the rest are
    reported as excluded so a changing denominator is visible.
    """
    ref = {row["sample_id"]: row for row in reference}
    cand = {row["sample_id"]: row for row in candidate}
    paired = []
    excluded = 0
    for key in sorted(set(ref) & set(cand)):
        left, right = ref[key], cand[key]
        if not (left.get("valid") and right.get("valid")
                and left.get("reference_category") and right.get("reference_category")):
            excluded += 1
            continue
        paired.append((left, right))
    ref_hits = sum(1 for left, _ in paired
                   if left.get("category_canonical") == left["reference_category"])
    cand_hits = sum(1 for _, right in paired
                    if right.get("category_canonical") == right["reference_category"])
    return {
        "paired_samples": len(paired),
        "excluded_samples": excluded,
        "reference_correct": ref_hits,
        "candidate_correct": cand_hits,
        "net_gain": cand_hits - ref_hits,
        "net_gain_definition": "candidate category hits minus reference category hits on the "
                               "same paired, valid, labelled samples",
    }
