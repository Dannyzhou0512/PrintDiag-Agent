"""PrintDiag web demo: read-only device status + saved-model replay diagnosis.

The web layer only orchestrates existing components: DiagnosisWorkflow,
RuleBasedProvider, DeviceReads, BM25 RAG, V0.3 adapter, optional V2 policy and
safety review. It never calls the cloud model and never writes to the device.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Callable

from printdiag.agents.composite import EvidenceGroundedDraftProvider
from printdiag.agents.symptom_routing import RuleBasedProvider
from printdiag.agents.workflow import DiagnosisWorkflow
from printdiag.rag.structured import packaged_structured_index
from printdiag.reliability.v2.output_guard import guard_output
from printdiag.reporting.diagnosis_view import view_markdown
from printdiag.web.answers import live_answer
from printdiag.reporting.lora_report import adapt_sft_report
from printdiag.reporting.safety_review import public_safety
from printdiag.schemas.agent import DiagnosisRequest, ToolResult
from printdiag.schemas.sft_diagnosis import DiagnosisValidationError, validate_diagnosis
from printdiag.tools.device_reads import DeviceReads, simulator_device
from printdiag.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[3]
HARNESS_PATH = ROOT / "scripts" / "p10_2_fault_harness.py"
DEMO_DATA = Path(__file__).resolve().parent / "demo_data" / "public_fixtures.json"

_spec = importlib.util.spec_from_file_location("p10_2_fault_harness", HARNESS_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

DeviceFactory = Callable[[], DeviceReads]


@dataclass(frozen=True)
class DemoScenario:
    scenario_id: str
    title: str
    description: str
    symptom: str
    device_source: str  # LIVE_SIMULATOR | SYNTHETIC_FIXTURE
    fixture_id: str | None
    model_source: str
    raw_output: str
    expected_v2: str


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def default_scenarios() -> dict[str, DemoScenario]:
    data = _load_json(DEMO_DATA)
    limit = json.dumps(data["limit"], ensure_ascii=False)
    motion = json.dumps(data["motion"], ensure_ascii=False)
    return {
        "A": DemoScenario(
            "A", "空闲模拟器与限位假设（V2 对照）",
            "当本地模拟器空闲且限位读数一致时，V2 会降级合成草稿的限位假设。",
            "打印机当前是否空闲？限位状态是否正常？", "LIVE_SIMULATOR", None,
            "SAVED_MODEL_REPLAY", limit, "contradicted"),
        "B": DemoScenario(
            "B", "合成端限位异常（V2 保留有证据的假设）",
            "合成 Fixture 的端限位触发与几何状态不一致；模型输出限位假设，V2 应判 supported。",
            "X轴归零异常，怀疑限位开关异常", "SYNTHETIC_FIXTURE", "endstop_mismatch",
            "SAVED_MODEL_REPLAY", limit, "supported"),
        "C": DemoScenario(
            "C", "工具证据不足（不虚构故障确认）",
            "合成设备读取失败；模型提出运动故障假设。V2 应判 insufficient/hypothesis_only，报告降级。",
            "打印机无法移动", "SYNTHETIC_FIXTURE", "motion_read_failure",
            "SAVED_MODEL_REPLAY", motion, "insufficient"),
    }


def _fixture_device(fixture_id: str) -> DeviceReads:
    scenario = next(s for s in base.SCENARIOS if s.scenario_id == fixture_id)
    return base.LoopbackMockReads(base.handler_for(scenario))


def _decision_from_limitations(limitations) -> tuple[str | None, list[str]]:
    codes = []
    for item in limitations:
        detail = getattr(item, "detail", item.get("detail", "") if isinstance(item, dict) else "")
        marker = "V2 编码："
        if marker in detail:
            codes.append(detail.split(marker, 1)[1].strip())
    decision = None
    if any(code in ("ENDSTOP_MISMATCH_OBSERVED", "THERMAL_FAULT_OBSERVED") for code in codes):
        decision = "supported"
    elif any(code.startswith("NORMAL_") or code == "NO_MATCHING_ANOMALY" for code in codes):
        decision = "contradicted"
    elif any(code in ("NO_TOOL_RECEIPT", "DEVICE_BOUNDARY",
                      "IDLE_DOES_NOT_EVIDENCE_MOTION_FAULT") for code in codes):
        decision = "hypothesis_only"
    elif codes:
        decision = "insufficient"
    return decision, codes


def _rag_hits(report) -> list[dict]:
    hits = []
    for rank, ref in enumerate(report.evidence_refs, start=1):
        if ref.source != "curated_knowledge":
            continue
        data = ref.data
        hits.append({
            "document_id": data.document_id,
            "chunk_id": getattr(data, "chunk_id", None),
            "title": data.title,
            "source_title": data.title,
            "source_type": "curated_knowledge",
            "excerpt": data.excerpt,
            "source_reference": getattr(data, "source_reference", None),
            "retrieval_score": getattr(data, "retrieval_score", None),
            "retrieval_rank": rank,
            "knowledge_version": getattr(data, "revision", None),
        })
    return hits


def _live_refs(report) -> list[dict]:
    refs = []
    for ref in report.evidence_refs:
        if ref.source != "live_read":
            continue
        refs.append({
            "tool": ref.tool,
            "evidence_id": ref.evidence_id,
            "observed_at_utc": ref.observed_at_utc.isoformat() if ref.observed_at_utc else None,
            "data": ref.data.model_dump(mode="json") if hasattr(ref.data, "model_dump") else ref.data,
        })
    return refs


def run_scenario(scenario: DemoScenario, *, symptom: str, v2: bool,
                 device_factory: DeviceFactory | None = None) -> dict:
    device = (device_factory or simulator_device)() if scenario.device_source == "LIVE_SIMULATOR" \
        else _fixture_device(scenario.fixture_id or "")
    provider = EvidenceGroundedDraftProvider(
        RuleBasedProvider(), lambda request, history: scenario.raw_output,
        label="saved model replay (no GPU)")

    presentation: dict = {}

    def adapter(diagnosis_id, content, **kwargs):
        from printdiag.reliability.v2.config import V2Config
        return adapt_sft_report(diagnosis_id, content,
                                v2_config=V2Config(enabled=v2), presentation=presentation, **kwargs)

    workflow = DiagnosisWorkflow(provider, ToolRegistry(device, packaged_structured_index()),
                                 draft_adapter=adapter)
    started = time.perf_counter()
    report = asyncio.run(workflow.diagnose(DiagnosisRequest(symptom=symptom)))
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    guard = guard_output(scenario.raw_output, attempt_repair=True)
    decision, codes = _decision_from_limitations(report.limitations)
    simulator_read = scenario.device_source == "LIVE_SIMULATOR" and device.provenance == "simulator"
    published_report = report.model_dump(mode="json")
    if not simulator_read:
        # The fixture may exercise the live-receipt branch in-process. Keep those
        # evidence links, but publish them as synthetic and never as live receipts.
        for ref in published_report["evidence_refs"]:
            if ref["source"] == "live_read":
                ref["source"] = "mock"
    payload = {
        "meta": {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "description": scenario.description,
            "device_source": "LIVE_SIMULATOR" if simulator_read else "SYNTHETIC_FIXTURE",
            "model_source": scenario.model_source,
            "live_inference": False,
            "gpu_requests": 0,
            "v2_enabled": v2,
            "v2_decision": decision if v2 else None,
            "v2_codes": codes if v2 else [],
            "guard": guard.as_public_dict(),
            "safety": public_safety(scenario.raw_output),
            "rag_hits": _rag_hits(report),
            "device_evidence": _live_refs(report) if simulator_read else [],
            "latency_ms": latency_ms,
            "observed_at_utc": datetime.now(UTC).isoformat(),
        },
        "report": published_report,
    }
    payload["answer"] = live_answer(payload["report"], payload["meta"],
                                    question=symptom, presentation=presentation)
    payload["meta"]["fault_category"] = payload["answer"]["diagnosis_view"]["category"]
    return payload


def device_status(device_factory: DeviceFactory | None = None) -> dict:
    factory = device_factory or simulator_device
    source = "LIVE_SIMULATOR" if factory is simulator_device else "UNKNOWN"
    provenance = "simulator" if factory is simulator_device else None
    try:
        device = factory()
        source = {("mock", "mock"): "SYNTHETIC_FIXTURE",
                  ("live_read", "simulator"): "LIVE_SIMULATOR"}.get(
                      (device.source, device.provenance), "UNKNOWN")
        provenance = device.provenance if source != "UNKNOWN" else None
        if source == "UNKNOWN":
            raise ValueError("unsupported device provenance")
        deadline = time.monotonic() + 5.0
        state, state_issues = device.printer_state(deadline)
        endstops, endstop_issues = device.endstops(["x_min", "y_min", "z_min"], deadline)
    except Exception as exc:  # noqa: BLE001 - UI must show offline state
        return {"online": False, "device_connected": False, "source": source,
                "provenance": provenance, "observed_at_utc": None, "read_only": True,
                "error": type(exc).__name__}
    issues = [issue.code for issue in (*state_issues, *endstop_issues)]
    has_endstop_observation = any(
        value is not None
        for component in endstops.components
        for value in (component.triggered, component.geometric_triggered, component.enabled)
    )
    if state is None and not has_endstop_observation:
        return {"online": False, "device_connected": False, "source": source,
                "provenance": provenance, "observed_at_utc": None,
                "error": "DEVICE_UNAVAILABLE" if "DEVICE_UNAVAILABLE" in issues else "DEVICE_STATUS_UNAVAILABLE",
                "read_only": True, "issues": issues}
    observed = datetime.now(UTC).isoformat()
    return {
        "online": True,
        "device_connected": source == "LIVE_SIMULATOR",
        "source": source,
        "provenance": provenance,
        "observed_at_utc": observed,
        "temperature": {
            "hotend": state.hotend_temperature if state else None,
            "hotend_target": state.hotend_target_temperature if state else None,
            "bed": state.bed_temperature if state else None,
            "bed_target": state.bed_target_temperature if state else None,
            "ambient": state.hotend_ambient_temperature if state else None,
        },
        "position_mm": state.position_mm.model_dump() if state and state.position_mm else None,
        "idle": state.idle if state else None,
        "moves_queued": state.moves_queued if state else None,
        "thermal_fault_count": state.thermal_fault_count if state else None,
        "endstops": endstops.model_dump(mode="json"),
        "read_only": True,
        "issues": issues,
    }


def markdown_report(payload: dict) -> str:
    return view_markdown(payload["answer"]["diagnosis_view"])
