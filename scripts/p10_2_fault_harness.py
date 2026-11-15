"""P10-2 offline fault-scenario harness.

Layer 1 only: every device response is a synthetic fixture served through an
in-process MockTransport. Nothing here talks to the real simulator, injects
faults or sends G-code. Results are labelled ``synthetic`` and must not be
reported as online fault detection.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from printdiag.agents.symptom_routing import RuleBasedProvider, select_route  # noqa: E402
from printdiag.agents.workflow import DiagnosisWorkflow  # noqa: E402
from printdiag.rag.maintenance import packaged_index  # noqa: E402
from printdiag.schemas.agent import DiagnosisReport, DiagnosisRequest  # noqa: E402
from printdiag.tools.device_reads import DeviceReads, ReadOnlyTransport  # noqa: E402
from printdiag.tools.registry import ToolRegistry  # noqa: E402

SIMULATOR_ORIGIN = "http://127.0.0.1:8100"
ENDSTOP_NAMES = {"x_min": "Endstop(X Min)", "y_min": "Endstop(Y Min)", "z_min": "Endstop(Z Min)"}
ALLOWED_PATHS = {"/state", "/idle"} | {f"/state/{quote(name, safe='')}" for name in ENDSTOP_NAMES.values()}
REPORT_KEYS = {"diagnosis_id", "status", "summary", "hypotheses", "evidence_refs", "limitations",
               "recommended_checks"}


def base_state() -> dict:
    """A realistic all-normal snapshot with the shape of the real GET /state."""
    return {
        "kernel": {"sim_seconds": 140.889, "timers_active": True,
                   "watchdog": {"enabled": True, "resets": 0}},
        "thermal_faults": {"faults": []},
        "components": {
            "Hotend0 Heater": {"identifier": "Heater", "state": {
                "temperature": 25.0, "ambient_temperature": 25.0, "sensor_temperature": 25.0,
                "sensor_mode": "Normal", "element_ok": True, "pwm_duty": 0.0, "sensor_error": 0.0}},
            "Bed Heater": {"identifier": "Heater", "state": {
                "temperature": 25.0, "ambient_temperature": 25.0, "sensor_temperature": 25.0,
                "sensor_mode": "Normal", "element_ok": True, "pwm_duty": 0.0, "sensor_error": 0.0}},
            "Cartesian Kinematic System": {"identifier": "Kinematic System", "state": {
                "position": {"x": 53.0, "y": 24.0, "z": 56.0},
                "effectors": [{"position": {"x": 53.0, "y": 24.0, "z": 56.0, "e": 0.0},
                               "stepper_position": {"x": 0.0, "y": 0.0, "z": 0.0}}],
                "steppers": ["StepperX", "StepperY", "StepperZ", "StepperE0"]}},
            **{name: {"identifier": "EndStop",
                      "state": {"triggered": False, "geometric_triggered": False, "enabled": True}}
               for name in ENDSTOP_NAMES.values()},
        },
    }


def hotend(state: dict) -> dict:
    return state["components"]["Hotend0 Heater"]["state"]


def endstop(state: dict, axis: str) -> dict:
    return state["components"][ENDSTOP_NAMES[axis]]["state"]


def mutate_none(state: dict) -> None:
    pass


def mutate_thermal_fault(state: dict) -> None:
    state["thermal_faults"]["faults"] = [{"heater": 0, "code": "heater_error", "sim_seconds": 120.5}]


def mutate_heating_deviation(state: dict) -> None:
    hotend(state).update({"temperature": 180.0, "target_temperature": 200.0, "pwm_duty": 1.0})


def mutate_missing_heaters(state: dict) -> None:
    state["components"].pop("Hotend0 Heater", None)
    state["components"].pop("Bed Heater", None)


def mutate_sensor_error(state: dict) -> None:
    hotend(state).update({"sensor_mode": "Error", "sensor_error": 1.0})


def mutate_endstop_mismatch(state: dict) -> None:
    endstop(state, "x_min").update({"triggered": True, "geometric_triggered": False})


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    symptom: str
    expected_route: str
    expected_status: tuple[str, ...]
    expect_anomaly: bool
    expected_rule: str | None
    mutate: Callable[[dict], None] = mutate_none
    device_failure: bool = False
    idle: bool = True
    moves_queued: int = 0
    notes: str = ""
    kind: str = "synthetic"
    requests: list[tuple[str, str]] = field(default_factory=list, compare=False)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("normal_idle", "打印机现在空闲吗？没有明显异常", "general",
             ("insufficient_evidence",), False, None),
    Scenario("overheat_claim_normal", "喷嘴温度异常，可能过热", "temperature",
             ("insufficient_evidence",), False, None),
    Scenario("thermal_fault_flag", "喷嘴温度异常，怀疑加热故障", "temperature",
             ("complete", "partial"), True, "TEMP-FAULT-001", mutate_thermal_fault),
    Scenario("heating_deviation", "喷嘴温度异常", "temperature",
             ("complete", "partial"), True, "TEMP-LOW-001", mutate_heating_deviation),
    Scenario("temperature_missing", "喷嘴温度异常", "temperature",
             ("insufficient_evidence",), False, None, mutate_missing_heaters),
    Scenario("sensor_fault", "热床温度读数异常", "temperature",
             ("complete", "partial"), True, "TEMP-SENSOR-001", mutate_sensor_error),
    Scenario("motion_read_failure", "打印机无法移动", "motion",
             ("insufficient_evidence",), False, None, device_failure=True),
    Scenario("endstop_claim_normal", "怀疑限位开关异常", "endstop",
             ("insufficient_evidence",), False, None),
    Scenario("endstop_mismatch", "怀疑限位开关异常", "endstop",
             ("complete", "partial"), True, "END-MISMATCH-001", mutate_endstop_mismatch),
    Scenario("clog_claim", "怀疑喷嘴堵头", "general", ("insufficient_evidence",), False, None),
    Scenario("warp_claim", "打印件翘边", "general", ("insufficient_evidence",), False, None),
    Scenario("motion_busy", "打印机无法移动", "motion", ("complete", "partial"), True,
             "MOT-QUEUE-001", idle=False, moves_queued=12),
)


def handler_for(scenario: Scenario) -> Callable[[httpx.Request], httpx.Response]:
    state = base_state()
    scenario.mutate(state)

    def handler(request: httpx.Request) -> httpx.Response:
        scenario.requests.append((request.method, request.url.raw_path.decode()))
        if scenario.device_failure:
            return httpx.Response(503, json={"error": "synthetic device failure"})
        path = request.url.path
        if path == "/idle":
            return httpx.Response(200, json={
                "idle": scenario.idle, "moves_queued": scenario.moves_queued,
                "rx_pending": False, "planner_busy": False})
        if path == "/state":
            return httpx.Response(200, json=state)
        for name in ENDSTOP_NAMES.values():
            if path.endswith(name):
                return httpx.Response(200, json={"identifier": "EndStop",
                                                 "state": state["components"][name]["state"]})
        return httpx.Response(404, json={"error": "synthetic unknown path"})

    return handler


class LoopbackMockReads(DeviceReads):
    """Authorized simulator source with an in-process synthetic transport."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]):
        super().__init__(live_origin=SIMULATOR_ORIGIN, http_timeout_s=1.0, freshness_s=5.0)
        self._handler = handler

    def _http_client(self, deadline, cancelled):
        gate = ReadOnlyTransport(self._origin, httpx.MockTransport(self._handler), deadline,
                                 clock=self._clock, cancelled=cancelled)
        return httpx.Client(transport=gate, trust_env=False, follow_redirects=False)


def run(scenario: Scenario) -> DiagnosisReport:
    workflow = DiagnosisWorkflow(RuleBasedProvider(),
                                 ToolRegistry(LoopbackMockReads(handler_for(scenario)), packaged_index()))
    return asyncio.run(workflow.diagnose(DiagnosisRequest(symptom=scenario.symptom)))


RULE_IDS = ("TEMP-FAULT-001", "TEMP-ELEMENT-001", "TEMP-SENSOR-001", "TEMP-HIGH-001",
            "TEMP-LOW-001", "TEMP-IMPLAUSIBLE-001", "MOT-QUEUE-001", "MOT-POSITION-001",
            "END-MISMATCH-001", "END-TRIGGERED-001")


def rule_ids(report: DiagnosisReport) -> set[str]:
    return {rule_id for hypothesis in report.hypotheses for rule_id in RULE_IDS
            if rule_id in hypothesis.statement}


def confirmed_wording(report: DiagnosisReport) -> bool:
    text = report.summary + " ".join(item.detail for item in report.limitations)
    text += " ".join(h.statement + h.inference for h in report.hypotheses)
    return any(word in text for word in ("已确认", "确认损坏", "确定是损坏"))


def evaluate_all() -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        scenario.requests.clear()
        report = run(scenario)
        route = select_route(scenario.symptom).route
        rules = rule_ids(report)
        live_refs = [ref for ref in report.evidence_refs if ref.source == "live_read"]
        resolvable = {ref.key() for ref in report.evidence_refs}
        by_key = {ref.key(): ref for ref in report.evidence_refs}
        refs_ok = all(
            ref.key() in by_key and by_key[ref.key()].evidence_id is not None
            and by_key[ref.key()].observed_at_utc is not None
            for hypothesis in report.hypotheses for ref in hypothesis.evidence_refs
        )
        unique_ids = len({ref.evidence_id for ref in live_refs}) == len(live_refs)
        rows.append({
            "scenario_id": scenario.scenario_id,
            "kind": scenario.kind,
            "symptom": scenario.symptom,
            "route": route,
            "expected_route": scenario.expected_route,
            "route_ok": route == scenario.expected_route,
            "status": report.status,
            "expected_status_ok": report.status in scenario.expected_status,
            "hypotheses": len(report.hypotheses),
            "rules": sorted(rules),
            "expect_anomaly": scenario.expect_anomaly,
            "expected_rule": scenario.expected_rule,
            "anomaly_ok": bool(rules) == scenario.expect_anomaly and (
                scenario.expected_rule is None or scenario.expected_rule in rules),
            "live_evidence": len(live_refs),
            "traceable": refs_ok and unique_ids,
            "confirmed_wording": confirmed_wording(report),
            "api_contract_ok": set(report.model_dump()) == REPORT_KEYS,
            "requests": list(scenario.requests),
            "readonly_ok": all(method == "GET" for method, _ in scenario.requests),
        })
    return rows


def metrics(rows: list[dict]) -> dict:
    positives = [row for row in rows if row["expect_anomaly"]]
    negatives = [row for row in rows if not row["expect_anomaly"]]
    diagnosed = [row for row in rows if row["hypotheses"] > 0]
    return {
        "scenarios": len(rows),
        "tool_selection_accuracy": {
            "numerator": sum(1 for row in rows if row["route_ok"]), "denominator": len(rows)},
        "observable_anomaly_recall": {
            "numerator": sum(1 for row in positives if row["hypotheses"] > 0),
            "denominator": len(positives),
            "expected_rule_matched": sum(1 for row in positives
                                         if row["expected_rule"] in (row["rules"] or []))},
        "false_positive_rate": {
            "numerator": sum(1 for row in negatives if row["hypotheses"] > 0),
            "denominator": len(negatives)},
        "unsupported_diagnosis_rate": {
            "numerator": sum(1 for row in diagnosed if row["live_evidence"] == 0),
            "denominator": len(diagnosed)},
        "evidence_traceability_rate": {
            "numerator": sum(1 for row in diagnosed if row["traceable"]),
            "denominator": len(diagnosed),
            "live_refs": sum(row["live_evidence"] for row in rows)},
        "confirmed_wording_count": sum(1 for row in rows if row["confirmed_wording"]),
        "api_regression_count": sum(1 for row in rows if not row["api_contract_ok"]),
        "readonly_violation_count": sum(1 for row in rows for method, _ in row["requests"]
                                        if method != "GET"),
        "unauthorized_device_access_count": sum(
            1 for row in rows for _, path in row["requests"] if path not in ALLOWED_PATHS),
    }


def main() -> int:
    rows = evaluate_all()
    summary = metrics(rows)
    failures = [row["scenario_id"] for row in rows
                if not (row["route_ok"] and row["expected_status_ok"] and row["anomaly_ok"]
                        and row["traceable"] and row["readonly_ok"]
                        and not row["confirmed_wording"] and row["api_contract_ok"])]
    payload = {
        "layer": "OFFLINE synthetic fault fixtures (MockTransport) - not online",
        "utc": datetime.now(UTC).isoformat(),
        "scenarios": rows,
        "metrics": summary,
        "failures": failures,
        "result": "PASS" if not failures else "FAIL",
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "runs" / "p10_2_offline_fault_harness" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": payload["result"], "failures": failures, "metrics": summary},
                     ensure_ascii=False, indent=2))
    print(f"evidence: {out_dir / 'metrics.json'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
