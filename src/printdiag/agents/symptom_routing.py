"""Deterministic, rule-based symptom routing for the read-only simulator mode.

This is **not** an LLM decision. It maps fixed keywords to a fixed set of
read-only tools and then drafts a report strictly from the ``ToolResult``
objects produced by this diagnosis.  Evidence references are only the
executor-issued ``evidence_id`` values; nothing is fabricated from the symptom
text, and no hypothesis is produced unless the observed readings show an
anomaly that the knowledge base can support.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Literal

from printdiag.agents.anomaly_rules import AnomalyThresholds, evaluate, select_primary
from printdiag.agents.providers import Exchange
from printdiag.reporting.agent_report import Check, ReportDraft
from printdiag.schemas.agent import (
    Axis, DiagnosisRequest, EndstopData, EvidenceKey, Hypothesis, KnowledgeData,
    PrinterStateProjection, ToolCallRequest, ToolResult,
)

RouteName = Literal["endstop", "motion", "temperature", "printing_stop", "general"]

_ENDPOINT = re.compile(r"限位|endstop|归零|trigger(?:ed)?|限位开关", re.I)
_MOTION = re.compile(r"无法移动|不能移动|不移动|卡住|运动|步进|电机|move|motion|jog|轴不动|停顿", re.I)
_TEMPERATURE = re.compile(r"温度|过热|加热|热床|热端|喷头温度|therm|heat|thermal", re.I)
_STOP = re.compile(r"突然停止|中途停止|停止打印|暂停|死机|halt|停住|中断|stopp?ed", re.I)
_AXIS_PATTERNS: tuple[tuple[Axis, re.Pattern[str]], ...] = (
    ("x_min", re.compile(r"x\s*(?:轴|min|_min)|x\s*限位", re.I)),
    ("y_min", re.compile(r"y\s*(?:轴|min|_min)|y\s*限位", re.I)),
    ("z_min", re.compile(r"z\s*(?:轴|min|_min)|z\s*限位", re.I)),
)

ROUTE_NOTE = "工具选择由确定性规则路由（rule-based），不是 LLM 自主决策。"
ALL_AXES: tuple[Axis, ...] = ("x_min", "y_min", "z_min")

# Knowledge lookup terms per route. The corpus ships one fragment per observation
# family (endstop / device-state / thermal); route-specific terms keep the hit
# relevant without turning the query into a free-text model judgement.
_KNOWLEDGE_TERMS: dict[RouteName, str] = {
    "endstop": "限位 触发 几何 安装 接线",
    "motion": "idle moves_queued 位置 运动 队列",
    "printing_stop": "idle moves_queued 位置 运动 队列 watchdog",
    "temperature": "温度 传感器 加热器 热故障",
    "general": "设备状态 观测 队列",
}
_KNOWLEDGE_CATEGORY: dict[RouteName, str | None] = {
    "endstop": "限位开关异常",
    "motion": "运动队列异常",
    "printing_stop": "设备无法启动或打印中断",
    "temperature": "喷嘴温度异常",
    "general": None,
}


def knowledge_query(plan: "RoutePlan", symptom: str) -> str:
    return f"{symptom[:200]} {_KNOWLEDGE_TERMS[plan.route]}".strip()[:300]


@dataclass(frozen=True)
class RoutePlan:
    route: RouteName
    device_calls: tuple[tuple[str, dict[str, object]], ...]
    checks: tuple[Check, ...]
    rationale: str


def _axes(symptom: str) -> tuple[Axis, ...]:
    found = tuple(axis for axis, pattern in _AXIS_PATTERNS if pattern.search(symptom))
    return found or ALL_AXES


def select_route(symptom: str) -> RoutePlan:
    """Map one symptom string to a fixed read-only tool plan (documented rules)."""
    if _STOP.search(symptom):
        return RoutePlan(
            "printing_stop", (("read_printer_state", {}),),
            ("review_observations", "request_details"),
            "症状含“停止/中断”类词，优先读取设备状态与队列。",
        )
    if _ENDPOINT.search(symptom):
        axes = _axes(symptom)
        return RoutePlan(
            "endstop",
            (("read_endstop_components", {"axes": list(axes)}), ("read_printer_state", {})),
            ("inspect_mounting", "inspect_cable_power_off"),
            "症状含“限位/归零”类词，优先读取限位组件与设备状态。",
        )
    if _MOTION.search(symptom):
        return RoutePlan(
            "motion",
            (("read_printer_state", {}), ("read_endstop_components", {"axes": list(ALL_AXES)})),
            ("review_observations",),
            "症状含“移动/运动”类词，优先读取运动队列、空闲状态与轴组件。",
        )
    if _TEMPERATURE.search(symptom):
        return RoutePlan(
            "temperature", (("read_printer_state", {}),),
            ("review_observations", "request_details"),
            "症状含“温度/加热”类词，优先读取设备状态与热故障计数。",
        )
    return RoutePlan(
        "general", (("read_printer_state", {}),),
        ("request_details",),
        "未命中专用规则，读取设备状态作为通用只读观测。",
    )


def _optional(value: float | None, unit: str = "") -> str:
    return "unavailable" if value is None else f"{value:g}{unit}"


def observation_summary(results: Sequence[ToolResult]) -> str:
    """Public, read-only rendering of this diagnosis's device observations."""
    return _facts(results)


def knowledge_hits(results: Sequence[ToolResult]) -> list:
    """Knowledge citations from this diagnosis, for prompt context only."""
    hits = []
    for result in results:
        if isinstance(result.data, KnowledgeData):
            hits.extend(result.data.hits)
    return hits


def _facts(results: Sequence[ToolResult]) -> str:
    parts: list[str] = []
    for result in results:
        data = result.data
        if isinstance(data, PrinterStateProjection):
            facts = []
            if data.idle is not None:
                facts.append(f"idle={str(data.idle).lower()}")
            if data.moves_queued is not None:
                facts.append(f"moves_queued={data.moves_queued}")
            if data.watchdog_resets is not None:
                facts.append(f"watchdog_resets={data.watchdog_resets}")
            if data.thermal_fault_count is not None:
                facts.append(f"thermal_fault_count={data.thermal_fault_count}")
            if data.position_mm is not None:
                position = data.position_mm
                facts.append(f"position_mm=({position.x:.1f},{position.y:.1f},{position.z:.1f})")
            if facts:
                parts.append("状态：" + "，".join(facts))
            temperature = []
            if data.hotend_temperature is not None or data.hotend_target_temperature is not None:
                temperature.append(
                    f"喷嘴当前={_optional(data.hotend_temperature, 'C')}，"
                    f"目标={_optional(data.hotend_target_temperature, 'C')}，"
                    f"环境={_optional(data.hotend_ambient_temperature, 'C')}，"
                    f"pwm={_optional(data.hotend_pwm_duty)}"
                )
            if data.bed_temperature is not None:
                temperature.append(
                    f"热床当前={_optional(data.bed_temperature, 'C')}，"
                    f"目标={_optional(data.bed_target_temperature, 'C')}"
                )
            if temperature:
                parts.append("温度：" + "；".join(temperature))
        elif isinstance(data, EndstopData):
            rows = [
                f"{row.axis} triggered={str(row.triggered).lower()} geometry={str(row.geometric_triggered).lower()}"
                for row in data.components
            ]
            if rows:
                parts.append("限位：" + "；".join(rows))
    return ("；".join(parts))[:420]


class RuleBasedProvider:
    """A documented deterministic policy: route → read-only tools → evidence draft."""

    mock = False

    def __init__(self, thresholds: AnomalyThresholds | None = None) -> None:
        self._thresholds = thresholds or AnomalyThresholds()

    async def next_turn(self, request: DiagnosisRequest, tools: list[dict],
                        history: Sequence[Exchange]) -> list[ToolCallRequest] | ReportDraft:
        plan = select_route(request.symptom)
        if not history:
            return [
                ToolCallRequest(call_id=f"route_{index}", tool=tool, arguments=dict(arguments))
                for index, (tool, arguments) in enumerate(plan.device_calls)
            ]
        if len(history) == 1:
            arguments: dict[str, object] = {"query": knowledge_query(plan, request.symptom)}
            category = _KNOWLEDGE_CATEGORY[plan.route]
            if category is not None:
                arguments["category"] = category
            return [ToolCallRequest(call_id="route_knowledge", tool="search_maintenance_knowledge",
                                    arguments=arguments)]
        return self._draft(request, plan, history)

    def notes_for(self, results: Sequence[ToolResult]) -> tuple[str, ...]:
        """Per-diagnosis limitations, computed from this run only."""
        facts = _facts(results)
        notes = [ROUTE_NOTE]
        if facts:
            notes.append(f"本轮设备观测：{facts}")
        notes.append("温度投影包含喷嘴/热床当前温度、环境温度、加热输出与传感器标志；"
                     "当前模拟器构建不提供目标温度字段，目标值显示为 unavailable，不做推测。")
        if any(result.status == "error" for result in results):
            notes.append("部分只读工具执行失败，相关字段不可用；未补造任何观测。")
        return tuple(notes)

    @staticmethod
    def _device_keys(results: Sequence[ToolResult]) -> list[EvidenceKey]:
        return [EvidenceKey(evidence_id=result.evidence_id) for result in results
                if result.evidence_id is not None and not isinstance(result.data, KnowledgeData)]

    @staticmethod
    def _knowledge_keys(results: Sequence[ToolResult]) -> list[EvidenceKey]:
        keys: list[EvidenceKey] = []
        for result in results:
            if result.evidence_id is not None and isinstance(result.data, KnowledgeData):
                keys.extend(EvidenceKey(evidence_id=result.evidence_id, hit_index=index)
                            for index in range(len(result.data.hits)))
        return keys

    def _draft(self, request: DiagnosisRequest, plan: RoutePlan,
               history: Sequence[Exchange]) -> ReportDraft:
        results = [result for exchange in history for result in exchange.results]
        device_keys = self._device_keys(results)
        knowledge_keys = self._knowledge_keys(results)
        hypothesis = None
        if device_keys and knowledge_keys:
            finding = select_primary(plan.route, evaluate(plan.route, results, self._thresholds))
            if finding is not None:
                statement = f"{finding.observed_abnormality}（{finding.rule_id}，待核查）"
                inference = (f"可能原因：{finding.possible_cause}。"
                             f"本轮观测：{_facts(results)}")
                hypothesis = Hypothesis(
                    statement=statement[:500], inference=inference[:500],
                    evidence_refs=[device_keys[0], knowledge_keys[0]],
                )
        summary = f"按症状规则选择只读工具（{plan.route}）。"[:490]
        return ReportDraft(
            summary=summary,
            hypotheses=[hypothesis] if hypothesis is not None else [],
            evidence_refs=(device_keys + knowledge_keys)[:30],
            limitations=[],
            recommended_checks=list(plan.checks),
        )
