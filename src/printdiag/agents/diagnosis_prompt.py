"""Training-format diagnosis prompts for the local LoRA adaptation.

The fine-tune used system/user/assistant messages with the LLaMA-Factory
``qwen3_5`` template and ``enable_thinking=false``. This module reproduces that
exact system prompt and the user-prompt shape, then appends this diagnosis's
device observations and (optionally) BM25 knowledge. The appended context is a
documented distribution shift: it was not part of the training inputs.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

DATASET_RELATIVE = ("dataset_SFT", "PrintDiag_Qwen3_4B_SFT_V02", "PrintDiag_SFT_V02_Ready",
                    "printdiag_qwen3_4b_sft_test.jsonl")
CUTOFF_LEN = 1024

# Runtime contract appended to the verbatim training system prompt. It states the
# frozen V0.3 seven-field schema, the two legal diagnosis_status values and the
# trustworthy-executor semantics of tool_observation_available. It deliberately
# contains no example answer and no reference category, so it cannot leak labels.
SEVEN_FIELD_CONTRACT = """输出契约（必须严格遵守）：
只输出一个合法 JSON 对象；不要 Markdown 代码块、不要解释、不要任何额外字段。
对象必须且只能包含以下 7 个字段：
- fault_category：字符串，故障类别（中文规范名称）
- suspected_cause：字符串，可能的故障原因（待验证假设，不能写成已确认结论）
- supporting_observation：字符串，支持判断的现象或线索，必须来自用户描述或本轮可信设备观测；没有可用依据时写"证据不足"
- recommended_checks：恰好 2 个互不相同的字符串组成的数组，给出可执行且安全的检查建议
- diagnosis_status：字符串，只能是 "待验证" 或 "证据不足" 之一，不得使用其他措辞
- tool_observation_available：JSON 布尔值 true/false，不得用字符串；只有现场线索明确来自本轮可信执行器工具回执时才可为 true，否则必须为 false，禁止虚构工具调用
- safety_note：字符串，涉及高温、运动或电气检查时必须强调先断电、冷却并按设备手册操作
禁止：输出英文嵌套结构；输出 evidence_id；声称已连接打印机或已执行工具；虚构温度、位置、限位或目标温度；建议用手触摸高温部件、带电拆卸或在高温状态下拆卸喷嘴。"""
SEVEN_FIELD_CONTRACT_SHA256 = sha256(SEVEN_FIELD_CONTRACT.encode("utf-8")).hexdigest()
DIAGNOSIS_STATUS_VALUES = ("待验证", "证据不足")


class PromptTemplateError(RuntimeError):
    """The training prompt template cannot be reproduced from local artifacts."""


def _project_root_with(relative: tuple[str, ...]) -> Path:
    root = Path(__file__).resolve().parents[3]
    for candidate in (root, *root.parents[:3]):
        if (candidate.joinpath(*relative)).is_file():
            return candidate
    return root


def dataset_path() -> Path:
    return _project_root_with(DATASET_RELATIVE).joinpath(*DATASET_RELATIVE)


@dataclass(frozen=True)
class TrainingTemplate:
    system_prompt: str
    example_user: str
    example_assistant: str
    source_path: str
    source_sha256: str
    dataset_rows: int

    def as_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "example_user": self.example_user,
            "example_assistant_excerpt": self.example_assistant[:400],
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "dataset_rows": self.dataset_rows,
            "template": "qwen3_5",
            "enable_thinking": False,
            "cutoff_len": CUTOFF_LEN,
        }


def load_training_template(path: Path | None = None) -> TrainingTemplate:
    """Read the exact system prompt used for fine-tuning (never invented)."""
    source = path or dataset_path()
    if not source.is_file():
        raise PromptTemplateError(f"training dataset not found: {source}")
    raw = source.read_bytes()
    lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not lines:
        raise PromptTemplateError("training dataset is empty")
    first = json.loads(lines[0])
    messages = first.get("messages")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
            "system", "user", "assistant"]:
        raise PromptTemplateError("training dataset does not use system/user/assistant messages")
    return TrainingTemplate(
        system_prompt=str(messages[0]["content"]),
        example_user=str(messages[1]["content"]),
        example_assistant=str(messages[2]["content"]),
        source_path=str(source),
        source_sha256=sha256(raw).hexdigest(),
        dataset_rows=len(lines),
    )


def _knowledge_block(knowledge: list[dict[str, Any]]) -> str:
    parts = []
    for item in knowledge[:3]:
        document_id = item.get("document_id", "?")
        chunk_id = item.get("chunk_id") or item.get("anchor") or "?"
        excerpt = str(item.get("excerpt", ""))[:300]
        reference = item.get("source_reference") or ""
        parts.append(f"[{document_id}/{chunk_id}] {excerpt}（来源：{reference}）")
    return "；".join(parts)


def build_diagnosis_messages(symptom: str, device_facts: str, knowledge: list[dict[str, Any]],
                             *, include_rag: bool, template: TrainingTemplate | None = None,
                             include_contract: bool = True
                             ) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return chat messages plus a prompt audit record.

    The system message keeps the fine-tune's system prompt verbatim and appends
    the frozen seven-field contract. The user line keeps the training shape
    ``3D打印故障记录：现象=…；现场线索=…。`` plus optional knowledge. Both
    additions are marked as out-of-distribution context in the audit.
    """
    tmpl = template or load_training_template()
    hints = device_facts.strip() or "无可用设备观测"
    user = f"3D打印故障记录：现象={symptom[:600]}；现场线索={hints[:900]}"
    if include_rag and knowledge:
        user += f"；维修知识参考={_knowledge_block(knowledge)}"
    user = user[:1800] + "。请不要将故障假设说成已验证结论，按指定JSON格式答复。"
    system = tmpl.system_prompt
    if include_contract:
        system = system + "\n\n" + SEVEN_FIELD_CONTRACT
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    audit = prompt_audit(tmpl, messages, symptom=symptom, device_facts=hints,
                         knowledge=knowledge, include_rag=include_rag,
                         include_contract=include_contract)
    return messages, audit


def prompt_audit(template: TrainingTemplate, messages: list[dict[str, str]], *, symptom: str,
                 device_facts: str, knowledge: list[dict[str, Any]], include_rag: bool,
                 include_contract: bool = True) -> dict:
    user = messages[-1]["content"]
    total_chars = sum(len(item["content"]) for item in messages)
    return {
        "template": "qwen3_5",
        "template_source": template.source_path,
        "template_sha256": template.source_sha256,
        "system_prompt_sha256": sha256(template.system_prompt.encode("utf-8")).hexdigest(),
        "system_prompt_chars": len(template.system_prompt),
        "schema_contract_included": include_contract,
        "schema_contract_sha256": SEVEN_FIELD_CONTRACT_SHA256 if include_contract else None,
        "diagnosis_status_enum": list(DIAGNOSIS_STATUS_VALUES),
        "tool_observation_semantics": "true only for this run's trusted executor receipts; "
                                      "otherwise false (JSON boolean)",
        "user_prompt_chars": len(user),
        "total_prompt_chars": total_chars,
        "estimated_tokens": total_chars // 2,  # rough CJK estimate, clearly labelled
        "estimated_tokens_note": "chars//2 heuristic; exact tokenizer unavailable",
        "cutoff_len": CUTOFF_LEN,
        "within_cutoff_estimate": total_chars // 2 <= CUTOFF_LEN,
        "device_context_chars": len(device_facts),
        "include_rag": include_rag,
        "knowledge_chunks": len(knowledge[:3]) if include_rag else 0,
        "distribution_shift": [
            "runtime device observations are appended to 现场线索; training samples used user-provided clues",
            "BM25 knowledge excerpts are appended when RAG is enabled; not present in training inputs",
            "the seven-field schema contract is appended to the training system prompt at runtime",
            "symptom text is free-form and may differ from dataset phrasing",
        ],
    }
