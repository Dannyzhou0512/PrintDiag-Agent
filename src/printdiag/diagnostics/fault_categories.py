"""Frozen SFT taxonomy sourced solely from the local train and validation splits."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import unicodedata

from printdiag.schemas.sft_diagnosis import parse_diagnosis_json, validate_diagnosis

INSUFFICIENT_CATEGORY = "证据不足或正常状态"

# Exact aliases reviewed against the meanings of train/validation labels.
# Keep neighboring categories separate, particularly global/local layer shift and kinds of thermal deformation.
_ALIASES: dict[str, tuple[str, ...]] = {
    "侧面条纹": ("侧壁条纹",),
    "模型倾斜": (),
    "尺寸精度异常": ("尺寸偏差",),
    "局部错层": ("局部层偏移",),
    "局部过热变形": ("局部过热导致变形",),
    "层错位": ("层移", "层移位", "层偏移"),
    "层间开裂": ("层间裂缝", "层间裂开"),
    "打印中途停止挤出": ("打印中途停止出料", "打印中途不出料"),
    "打印头越界": ("喷头越界",),
    "拉丝或渗料": ("拉丝", "渗料", "拉丝与渗料"),
    "挤出不足": ("欠挤出", "挤出量不足"),
    "挤出不稳定": ("挤出不均匀", "挤出忽多忽少"),
    "振纹与振铃": ("振铃", "振纹", "振纹或振铃"),
    "填充与外壁存在缝隙": ("填充和外壁之间有缝隙",),
    "填充强度不足": ("填充强度低",),
    "外壁透出填充纹理": ("外壁可见填充纹理",),
    "桥接质量差": ("桥接不良",),
    "模型翘曲": ("模型翘边",),
    "模型难以脱离平台": ("模型难以从平台取下",),
    "热变形": ("受热变形",),
    "缺失打印层": ("缺层", "打印层缺失"),
    "细小特征无法打印": ("细小特征打印不出",),
    "耗材断裂": ("耗材折断",),
    "耗材被挤出齿轮磨损": ("挤出齿轮磨损耗材",),
    "薄壁缺口": ("薄壁有缺口",),
    "支撑上方表面粗糙": ("支撑上方表面不平整",),
    "表面斜向划痕": ("表面斜向刮痕",),
    "表面疙瘩与接缝": ("表面疙瘩和接缝",),
    "象脚效应": ("象脚",),
    "喷嘴堵塞": ("喷嘴阻塞", "喷嘴堵住"),
    "起始不出丝": ("起始不出料", "开始打印不出丝"),
    "顶层缝隙": ("顶层间隙",),
    "顶层鼓包与孔洞": ("顶层鼓包和孔洞",),
    "顶面划痕": ("顶面刮痕",),
    "首层不粘附": ("首层不粘", "首层不粘床", "首层附着不良"),
    "首层质量异常": ("首层质量问题",),
    INSUFFICIENT_CATEGORY: ("证据不足", "信息不足", "正常状态"),
}

CANONICAL_CATEGORIES = frozenset(_ALIASES)


def _key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _build_index() -> dict[str, str]:
    index = {}
    for canonical, aliases in _ALIASES.items():
        for name in (canonical, *aliases):
            key = _key(name)
            if key in index:
                raise ValueError(f"duplicate normalized category name: {name}")
            index[key] = canonical
    return index


_INDEX = _build_index()


@dataclass(frozen=True, slots=True)
class CategoryMatch:
    raw: str
    canonical: str | None
    match: Literal["canonical", "alias", "unmapped"]


def normalize_fault_category(label: str) -> CategoryMatch:
    if type(label) is not str or not label.strip():
        raise ValueError("fault category must be a nonempty string")
    key = _key(label)
    canonical = _INDEX.get(key)
    if canonical is None:
        return CategoryMatch(label, None, "unmapped")
    return CategoryMatch(label, canonical, "canonical" if key == _key(canonical) else "alias")


def extract_training_categories(dataset_dir: Path) -> dict[str, frozenset[str]]:
    """Read only the two named development splits. Never open the test split."""
    result = {}
    for split in ("train", "val"):
        categories = set()
        path = dataset_dir / f"printdiag_qwen3_4b_sft_{split}.jsonl"
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = parse_diagnosis_json(line)
                    messages = row["messages"]
                    if not isinstance(messages, list) or [item["role"] for item in messages] != [
                        "system", "user", "assistant",
                    ] or type(messages[-1]["content"]) is not str:
                        raise ValueError("invalid messages")
                    categories.add(validate_diagnosis(messages[-1]["content"]).fault_category)
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"{split} row {number}: invalid development example") from error
        if not categories:
            raise ValueError(f"{split}: no examples")
        result[split] = frozenset(categories)
    return result
