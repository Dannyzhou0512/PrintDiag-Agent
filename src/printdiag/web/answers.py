"""User-readable answers and formatting; no diagnosis logic is duplicated here."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from printdiag.reporting.diagnosis_view import make_view
from printdiag.web.intent import Intent, IntentRoute, route_intent


def format_local_time(value: str | None, *, tz: ZoneInfo | None = None) -> str:
    if not value:
        return "未知"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "未知"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    local = parsed.astimezone(tz or datetime.now().astimezone().tzinfo)
    return local.strftime("%Y-%m-%d %H:%M:%S")


def format_coordinate(position) -> str:
    if not position:
        return "未知"
    if hasattr(position, "model_dump"):
        position = position.model_dump()
    if not isinstance(position, dict):
        return "未知"
    parts = []
    for axis in ("x", "y", "z"):
        value = position.get(axis)
        if value is None:
            parts.append(f"{axis.upper()} 未知")
        else:
            parts.append(f"{axis.upper()} {float(value):.1f} mm")
    return "、".join(parts)


def _temperature(value) -> str:
    return "未知" if value is None else f"{float(value):.0f}°C"


def is_device_status_query(text: str) -> bool:
    return route_intent(text).intent is Intent.DEVICE_STATUS


def intent_answer(route: IntentRoute, live_status: dict | None = None) -> dict:
    if route.intent is Intent.GREETING:
        return {"mode": "CHAT_GREETING", "title": "你好",
                "conclusion": ("你好！我是 PrintDiag-Agent，可以查看打印机实时状态，"
                               "也可以帮助分析打印过程中遇到的故障。你可以直接描述问题。"),
                "evidence": [], "possible_causes": [], "next_steps": [],
                "uncertain": []}
    if route.intent is Intent.MODEL_IDENTITY:
        models = (live_status or {}).get("models") or ["base", "lora"]
        mode_text = "、".join("Base" if item == "base" else "LoRA" for item in models)
        return {"mode": "MODEL_IDENTITY", "title": "模型说明",
                "conclusion": (f"我是 PrintDiag-Agent 智能诊断助手。故障诊断使用 Qwen3.5-4B，"
                               f"支持 {mode_text} 两种模式。设备状态由 Marlin Simulator 提供，"
                               "V2 负责核查诊断证据。"),
                "evidence": [f"可用模式：{mode_text}", "Qwen 不在问候/身份问题中调用"],
                "possible_causes": [], "next_steps": [], "uncertain": []}
    if route.intent is Intent.CAPABILITIES:
        return {"mode": "CAPABILITIES", "title": "我可以做什么",
                "conclusion": ("我可以查看打印机的实时状态（温度、位置、限位、空闲/故障计数），"
                               "也可以分析打印故障，结合设备证据与维修知识给出待验证判断和检查建议。"),
                "evidence": ["设备状态：只读工具直接回答，不消耗 GPU",
                             "故障诊断：Qwen3.5-4B Base/LoRA，经七字段校验与 V2 证据核查"],
                "possible_causes": [], "next_steps": ["描述具体症状或询问当前状态"],
                "uncertain": []}
    return {"mode": "CLARIFICATION_NEEDED", "title": "需要更多信息",
            "conclusion": ("我还不能判断具体问题。请描述打印机现象，例如“模型欠挤出”、"
                           "“喷嘴无法升温”或“首层不粘”，也可以先问“当前打印机是什么状态？”。"),
            "evidence": [], "possible_causes": [], "next_steps": ["补充具体故障现象或设备状态"],
            "uncertain": ["信息不足时不会猜测或制造故障结论"]}


def device_answer(status: dict, *, now: datetime | None = None) -> dict:
    if not status.get("online"):
        return {"mode": "DEVICE_STATUS_ONLY", "title": "设备状态",
                "conclusion": "设备当前离线，无法读取实时状态。请检查 Marlin Simulator（127.0.0.1:8100）后刷新。",
                "evidence": ["设备状态接口不可达"],
                "possible_causes": [], "next_steps": ["确认模拟器已启动", "稍后点击刷新"],
                "uncertain": ["离线状态下无法确认任何设备异常"], "status": status}
    temperature = status.get("temperature") or {}
    position = format_coordinate(status.get("position_mm"))
    endstops = (status.get("endstops") or {}).get("components") or []
    triggered = [item["axis"] for item in endstops if item.get("triggered") is True]
    idle = status.get("idle")
    hotend = _temperature(temperature.get("hotend"))
    bed = _temperature(temperature.get("bed"))
    observed = status.get("observed_at_utc")
    stale = False
    if observed:
        try:
            stamp = datetime.fromisoformat(observed.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            stale = ((now or datetime.now(UTC)) - stamp).total_seconds() > 30
        except ValueError:
            stale = False
    state_text = "空闲" if idle is True else ("正在执行任务" if idle is False else "状态未知")
    source_description = {
        "SYNTHETIC_FIXTURE": "以下为 Mock 示例，未连接设备",
        "LIVE_SIMULATOR": "Marlin Simulator 在线",
    }.get(status.get("source"), "设备读数来源未核实")
    conclusion = (
        f"{source_description}。当前读数显示{state_text}。"
        f"喷嘴温度 {hotend}，热床温度 {bed}，当前位置：{position}。"
        f"{'三轴限位当前未触发。' if not triggered else '触发限位：' + '、'.join(triggered) + '。'}"
        f"待执行运动 {status.get('moves_queued') if status.get('moves_queued') is not None else '未知'} 条，"
        f"热故障计数 {status.get('thermal_fault_count') if status.get('thermal_fault_count') is not None else '未知'}。"
    )
    if stale:
        conclusion += " 注意：最近一次采集可能已过期。"
    evidence = [
        f"喷嘴/热床/环境：{hotend} / {bed} / {_temperature(temperature.get('ambient'))}",
        f"坐标：{position}",
        f"限位：{'触发 ' + '、'.join(triggered) if triggered else '三轴未触发'}",
        f"采集时间：{format_local_time(observed)}",
    ]
    return {
        "mode": "DEVICE_STATUS_ONLY", "title": "设备当前状态",
        "conclusion": conclusion, "evidence": evidence,
        "possible_causes": [], "next_steps": ["如需故障分析，请描述具体症状并切换到 AI 诊断模式"],
        "uncertain": ["三轴限位未触发仅表示当前未触发，不代表所有限位开关功能正常",
                      "设备状态正常不代表不存在质量类问题"],
        "status": status,
    }


def live_answer(report: dict, meta: dict, *, question='', presentation=None) -> dict:
    status = report.get("status")
    decision = meta.get("v2_decision")
    hypotheses = report.get("hypotheses") or []
    safety = meta.get("safety") or {}
    if safety.get("status") == "review_required":
        conclusion = "维修建议的安全前置条件无法确认，已隐藏并标记为需人工复核。"
    elif safety.get("blocked"):
        conclusion = "模型输出包含危险维修建议，已被安全审查拦截；不能作为可执行方案。"
    elif decision in ("contradicted", "insufficient") or status == "insufficient_evidence":
        conclusion = ("当前证据不足，不能确认具体故障。以下仅为待核查方向，"
                      "不应把模型推测当作已验证结论。")
    elif hypotheses:
        conclusion = "基于用户描述与当前只读设备证据，存在待验证的故障假设（尚未确认）。"
    else:
        conclusion = "当前没有足够证据形成故障结论；建议补充症状或设备信息。"
    evidence = []
    for ref in meta.get("device_evidence") or []:
        evidence.append(f"{ref.get('tool')} · 采集于 {format_local_time(ref.get('observed_at_utc'))}")
    if not evidence:
        evidence.append("本轮没有可用的设备实时证据")
    evidence.extend(f"RAG 背景：{hit.get('document_id')}（不是设备观测）"
                    for hit in meta.get("rag_hits") or [])
    causes = [f"{item.get('statement')}（{item.get('inference')}）"
              for item in hypotheses if item.get("statement")]
    checks = report.get("recommended_checks") or []
    uncertain = [item.get("detail") for item in report.get("limitations") or []
                 if item.get("detail")]
    return {"mode": meta.get("mode"), "title": "AI 诊断建议",
            "conclusion": conclusion, "evidence": evidence,
            "possible_causes": causes, "next_steps": checks,
            "uncertain": uncertain, "model": meta.get("model"),
            "diagnosis_view": make_view(report, meta, question=question, presentation=presentation),
            "live_inference": meta.get("live_inference")}
