"""Deterministic anomaly rules over this diagnosis's projected device readings.

Three levels are kept strictly separate:

* ``observed_abnormality`` — what the device actually reported in this run;
* ``possible_cause`` — what a human still has to check, never a confirmed cause;
* ``confirmed_fault`` — always False: a single read-only snapshot cannot confirm a
  physical hardware fault, and the agent has no authority to do so.

Every rule lists the fields it needs. Missing fields never produce a finding:
they degrade to "insufficient evidence" instead of inventing values.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from printdiag.schemas.agent import EndstopData, PrinterStateProjection, ToolResult

AnomalyCategory = Literal["temperature", "motion", "endstop"]
Confidence = Literal["observed", "possible"]
RouteName = Literal["endstop", "motion", "temperature", "printing_stop", "general"]


@dataclass(frozen=True)
class AnomalyThresholds:
    """Engineering heuristics, not printer-industry standards.

    Both values are configurable constructor arguments of ``RuleBasedProvider``.
    ``temperature_deviation_c`` only decides whether a *possible* temperature
    deviation deserves an observation; it never confirms a fault.
    """

    temperature_deviation_c: float = 15.0
    implausible_vs_ambient_c: float = 80.0


@dataclass(frozen=True)
class AnomalyRule:
    rule_id: str
    category: AnomalyCategory
    routes: tuple[str, ...]  # ("*",) = always evaluated; otherwise route-gated
    required_fields: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class AnomalyFinding:
    rule_id: str
    category: AnomalyCategory
    confidence: Confidence
    observed_abnormality: str
    possible_cause: str
    confirmed_fault: bool = False


RULES: tuple[AnomalyRule, ...] = (
    AnomalyRule("TEMP-FAULT-001", "temperature", ("*",), ("thermal_fault_count",),
                "Firmware reported at least one thermal fault for this run."),
    AnomalyRule("TEMP-ELEMENT-001", "temperature", ("*",), ("hotend_element_ok", "bed_element_ok"),
                "A heater's element self-check reported not-ok."),
    AnomalyRule("TEMP-SENSOR-001", "temperature", ("*",),
                ("hotend_sensor_mode", "bed_sensor_mode", "hotend_sensor_error", "bed_sensor_error"),
                "A temperature sensor reported a non-normal mode or a nonzero error."),
    AnomalyRule("TEMP-HIGH-001", "temperature", ("temperature",), ("hotend_temperature", "hotend_target_temperature"),
                "Current temperature is above the configured target by a configured margin."),
    AnomalyRule("TEMP-LOW-001", "temperature", ("temperature",), ("hotend_temperature", "hotend_target_temperature"),
                "Current temperature is below the configured target by a configured margin."),
    AnomalyRule("TEMP-IMPLAUSIBLE-001", "temperature", ("temperature",),
                ("hotend_temperature", "hotend_ambient_temperature"),
                "Temperature is implausibly far above ambient for a read-only snapshot."),
    AnomalyRule("MOT-QUEUE-001", "motion", ("motion", "printing_stop"), ("idle", "moves_queued"),
                "The motion queue or planner is not idle."),
    AnomalyRule("MOT-POSITION-001", "motion", ("motion", "printing_stop"), ("position_mm",),
                "Position was missing while the user reports a movement problem."),
    AnomalyRule("END-MISMATCH-001", "endstop", ("endstop",),
                ("components",), "Effective trigger and geometric position disagree."),
    AnomalyRule("END-TRIGGERED-001", "endstop", ("endstop",), ("components",),
                "An endstop is triggered in a static snapshot."),
)


def _state(results: Sequence[ToolResult]) -> PrinterStateProjection | None:
    return next((r.data for r in results if isinstance(r.data, PrinterStateProjection)), None)


def _endstops(results: Sequence[ToolResult]) -> EndstopData | None:
    return next((r.data for r in results if isinstance(r.data, EndstopData)), None)


def _route_matches(rule: AnomalyRule, route: str) -> bool:
    return rule.routes == ("*",) or route in rule.routes


def evaluate(route: RouteName | str, results: Sequence[ToolResult],
             thresholds: AnomalyThresholds | None = None) -> tuple[AnomalyFinding, ...]:
    """Evaluate every applicable rule; return findings in deterministic order."""
    limits = thresholds or AnomalyThresholds()
    state = _state(results)
    endstops = _endstops(results)
    findings: list[AnomalyFinding] = []
    for rule in RULES:
        if not _route_matches(rule, route):
            continue
        finding = _evaluate_rule(rule, state, endstops, limits)
        if finding is not None:
            findings.append(finding)
    return tuple(findings)


def _evaluate_rule(rule: AnomalyRule, state: PrinterStateProjection | None,
                   endstops: EndstopData | None, limits: AnomalyThresholds) -> AnomalyFinding | None:
    if state is None and rule.category != "endstop":
        return None
    if rule.rule_id == "TEMP-FAULT-001" and state is not None:
        count = state.thermal_fault_count
        if count is not None and count > 0:
            return AnomalyFinding(rule.rule_id, "temperature", "observed",
                                  f"固件本轮回报了 {count} 个热故障",
                                  "加热回路或温度传感器需要核查（具体故障原因未确认）")
    if rule.rule_id == "TEMP-ELEMENT-001" and state is not None:
        failed = [name for name, ok in (("喷嘴", state.hotend_element_ok), ("热床", state.bed_element_ok))
                  if ok is False]
        if failed:
            return AnomalyFinding(rule.rule_id, "temperature", "observed",
                                  f"{'、'.join(failed)}加热元件自检未通过",
                                  "加热元件或其驱动电路需要核查（原因未确认）")
    if rule.rule_id == "TEMP-SENSOR-001" and state is not None:
        modes = [mode for mode in (state.hotend_sensor_mode, state.bed_sensor_mode)
                 if mode is not None and mode != "Normal"]
        errors = [value for value in (state.hotend_sensor_error, state.bed_sensor_error)
                  if value is not None and value != 0]
        if modes or errors:
            return AnomalyFinding(rule.rule_id, "temperature", "observed",
                                  "温度传感器报告了非正常模式或非零错误值",
                                  "温度传感器接线或读数需要核查（原因未确认）")
    if rule.rule_id == "TEMP-HIGH-001" and state is not None:
        temperature, target = state.hotend_temperature, state.hotend_target_temperature
        if temperature is not None and target is not None and target > 0 and temperature - target >= limits.temperature_deviation_c:
            return AnomalyFinding(rule.rule_id, "temperature", "possible",
                                  f"喷嘴当前温度 {temperature:g} °C 高于目标 {target:g} °C，偏差 {temperature - target:g} °C",
                                  "单次快照可能处于降温阶段，需要持续观测温度趋势后再判断")
    if rule.rule_id == "TEMP-LOW-001" and state is not None:
        temperature, target = state.hotend_temperature, state.hotend_target_temperature
        if temperature is not None and target is not None and target > 0 and target - temperature >= limits.temperature_deviation_c:
            return AnomalyFinding(rule.rule_id, "temperature", "possible",
                                  f"喷嘴当前温度 {temperature:g} °C 低于目标 {target:g} °C，偏差 {target - temperature:g} °C",
                                  "可能处于正常升温阶段或加热能力不足，需要观测趋势与加热输出")
    if rule.rule_id == "TEMP-IMPLAUSIBLE-001" and state is not None:
        temperature, ambient = state.hotend_temperature, state.hotend_ambient_temperature
        if (temperature is not None and ambient is not None
                and temperature - ambient >= limits.implausible_vs_ambient_c
                and (state.hotend_pwm_duty or 0) == 0):
            return AnomalyFinding(rule.rule_id, "temperature", "observed",
                                  f"喷嘴温度 {temperature:g} °C 远高于环境 {ambient:g} °C，但加热输出为 0",
                                  "温度读数或传感器可能异常，需要核查（原因未确认）")
    if rule.rule_id == "MOT-QUEUE-001" and state is not None:
        if state.idle is False or (state.moves_queued or 0) > 0:
            return AnomalyFinding(rule.rule_id, "motion", "observed",
                                  f"运动队列非空闲（idle={state.idle}, moves_queued={state.moves_queued}）",
                                  "设备可能仍在执行动作或队列未清空；这本身不等于电机故障")
    if rule.rule_id == "MOT-POSITION-001" and state is not None:
        if state.position_mm is None and state.idle is not False:
            return AnomalyFinding(rule.rule_id, "motion", "possible",
                                  "本轮未能取得位置投影，无法判断轴是否运动",
                                  "位置字段不可用，需要重新采样或人工检查（未确认故障）")
    if rule.rule_id == "END-MISMATCH-001" and endstops is not None:
        mismatched = [row.axis for row in endstops.components
                      if row.triggered is True and row.geometric_triggered is False]
        if mismatched:
            return AnomalyFinding(rule.rule_id, "endstop", "observed",
                                  f"限位有效触发与几何位置不一致：{', '.join(mismatched)}",
                                  "信号路径、安装条件或开关本身需要核查（原因未确认）")
    if rule.rule_id == "END-TRIGGERED-001" and endstops is not None:
        triggered = [row.axis for row in endstops.components if row.triggered is True]
        if triggered:
            return AnomalyFinding(rule.rule_id, "endstop", "observed",
                                  f"限位处于触发状态：{', '.join(triggered)}",
                                  "需结合几何位置判断是正常触发还是持续触发（原因未确认）")
    return None


def select_primary(route: RouteName | str, findings: Sequence[AnomalyFinding]) -> AnomalyFinding | None:
    """Prefer the finding that matches the routed symptom; fall back to the first."""
    for finding in findings:
        if finding.category == route:
            return finding
    return findings[0] if findings else None
