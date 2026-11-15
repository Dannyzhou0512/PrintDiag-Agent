"""Lightweight intent routing for the web assistant.

The router is deliberately rule-based and testable; it is the single place that
decides whether a message needs Qwen, the read-only device tools, or no work at
all. It can later be replaced by an LLM router without changing the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


class Intent(StrEnum):
    GREETING = "greeting"
    MODEL_IDENTITY = "model_identity"
    CAPABILITIES = "capabilities"
    DEVICE_STATUS = "device_status"
    FAULT_DIAGNOSIS = "fault_diagnosis"
    CLARIFICATION_NEEDED = "clarification_needed"


@dataclass(frozen=True)
class IntentRoute:
    intent: Intent
    reason: str
    matched: str | None = None


_GREETINGS = ("你好", "您好", "嗨", "哈喽", "hello", "hi", "早上好", "下午好", "晚上好")
_MODEL_TERMS = ("什么模型", "哪个模型", "你是谁", "你用的模型", "模型身份", "什么ai",
                "什么 ai", "大模型", "你是什么", "base", "lora", "qwen")
_CAPABILITY_TERMS = ("能做什么", "可以做什么", "会什么", "有什么功能", "功能", "怎么用",
                     "你能帮我", "帮助")
_FAULT_TERMS = ("欠挤出", "挤出不足", "挤出不够", "无法升温", "不升温", "升不了温", "温度异常",
                "喷嘴堵", "堵头", "堵塞", "拉丝", "渗料", "漏料", "翘边", "翘曲", "不粘",
                "首层", "层错位", "错层", "缺层", "缺失打印层", "断裂", "不出丝", "无法出丝",
                "异响", "振动", "振纹", "网格", "支撑", "变形", "粗糙", "气泡", "缝隙", "缺口",
                "尺寸", "精度", "外壁明显偏细", "过热", "打印失败", "报错", "故障", "冷却", "风扇", "堵料",
                "挤出轮", "送料", "打印出来", "打印件", "模型出现", "打印时", "异常")
_STATUS_TERMS = ("状态", "温度", "多少度", "几度", "坐标", "位置", "限位", "空闲", "idle",
                 "在打印", "打印中", "正常吗", "是否正常", "热床", "喷头温度", "喷嘴温度")
_CLARIFY_TERMS = ("帮我看看", "这是怎么回事", "怎么回事", "怎么办", "有问题", "看看",
                  "分析一下", "判断一下", "什么原因", "如何排查", "怎么排查")
_FAULT_QUESTION = ("无法", "不能", "不出", "异常", "故障", "问题", "失败", "报错", "怎么办",
                   "原因", "排查", "维修", "解决")


def _contains(text: str, terms: tuple[str, ...]) -> str | None:
    for term in terms:
        if term in text:
            return term
    return None


def route_intent(text: str) -> IntentRoute:
    normalized = re.sub(r"\s+", " ", text.strip().casefold())
    if not normalized:
        return IntentRoute(Intent.CLARIFICATION_NEEDED, "empty input")
    model = _contains(normalized, _MODEL_TERMS)
    fault = _contains(normalized, _FAULT_TERMS)
    if model and (not fault or model not in ("base", "lora", "qwen")):
        return IntentRoute(Intent.MODEL_IDENTITY, "model identity terms", model)
    if fault:
        return IntentRoute(Intent.FAULT_DIAGNOSIS, "fault symptom terms", fault)
    capability = _contains(normalized, _CAPABILITY_TERMS)
    if capability:
        return IntentRoute(Intent.CAPABILITIES, "capability terms", capability)
    greeting = _contains(normalized, _GREETINGS)
    status = _contains(normalized, _STATUS_TERMS)
    if status:
        return IntentRoute(Intent.DEVICE_STATUS, "device status terms", status)
    if greeting:
        # A bare greeting or "你好，你能做什么" already handled above; a greeting
        # without a concrete question must not enter the diagnosis pipeline.
        return IntentRoute(Intent.GREETING, "greeting only", greeting)
    clarify = _contains(normalized, _CLARIFY_TERMS)
    if clarify:
        return IntentRoute(Intent.CLARIFICATION_NEEDED, "vague request", clarify)
    if _contains(normalized, _FAULT_QUESTION) and len(normalized) <= 32:
        return IntentRoute(Intent.CLARIFICATION_NEEDED, "question needs a concrete symptom")
    if len(normalized) <= 8:
        return IntentRoute(Intent.CLARIFICATION_NEEDED, "too short to diagnose")
    return IntentRoute(Intent.CLARIFICATION_NEEDED, "no concrete symptom or status question")
