"""Review model-generated maintenance advice before publication.

The rules identify a bounded bilingual vocabulary of high-risk actions. Scope
markers separate prohibitions and historical examples from executable advice.
"""

from dataclasses import dataclass
import re
from typing import Iterable, Sequence
import unicodedata

from printdiag.schemas.sft_diagnosis import DiagnosisValidationError, SFTDiagnosisDraft, validate_diagnosis

SAFETY_REVIEW_VERSION = "p10-9a3-bilingual"
_REVIEW_REQUIRED = "UNVERIFIED_SERVICE_PRECONDITIONS"

# End strong clauses and restart intent after contrast, an explicit new step,
# or a permission. Commas alone may separate an action from its safety condition.
_SCOPE_BREAK = re.compile(
    r"[。！？!?；;\n]+|(?<!\d)\.(?!\d)"
    r"|但是|但|不过|然而|可是|而是|随后|然后|接着|下一步"
    r"|\b(?:but|however|yet|then|next)\b|\binstead\b(?!\s+of\b)"
    r"|\band\s+(?=(?:you\s+)?(?:may|can|should|must|do|go ahead)\b)"
    r"|[,，]\s*(?=(?:必须|务必|可以|仍可|直接|you (?:may|can|must|should)\b|go ahead\b))"
)
_FORBID = re.compile(
    r"严禁|禁止|切勿|请勿|勿|避免|不要|不得|不可|不能|不应|不准"
    r"|\b(?:do not|don't|never|must not|should not|cannot|can't|avoid|refrain from|instead of|rather than"
    r"|it is (?:unsafe|dangerous|prohibited|forbidden) to)\b"
)
_FALSE_NEGATION = re.compile(
    r"^\s*(?:忘记|忘了|犹豫|假设|认为|(?:forget|hesitate|fail)\s+to\b|assume\b)"
)
_EXCEPTION = re.compile(r"除非|例外情况|\b(?:unless|except\s+(?:when|if))\b")
_PERMISSION = re.compile(r"^(?:[^。；;]{0,20})?(?:仍可|可以|允许|\b(?:you may|go ahead)\b)")
_QUOTE = re.compile(r"“[^”]+”|「[^」]+」|\"[^\"]+\"|(?<!\w)'[^'\r\n]+'(?!\w)")
_REPORTS = re.compile(r"日志|记录|报告|手册|培训|教材|\b(?:incident|log|manual|transcript|training)\b")
_PROHIBITED_EXAMPLE = re.compile(
    r"列(?:作|为)|禁止操作|违规操作|危险示例|发生的操作记录"
    r"|\b(?:prohibited|forbidden|unsafe)\s+(?:example|action|practice)\b"
    r"|\b(?:action|procedure) (?:recorded|observed|reported) (?:yesterday|earlier)\b"
)
_DISCUSSION = re.compile(r"^\s*(?:讨论为何|说明为何|\b(?:discuss|explain) why\b)")
_RISK_DESCRIPTION = re.compile(r"很危险|有风险|属于违规|\b(?:dangerous|unsafe|hazardous)\b")
_RISK_VERDICT = re.compile(
    r"^\s*(?:(?:是|属于)(?:危险|违规|禁止)|(?:is|would be)\s+(?:dangerous|unsafe|forbidden|prohibited)\b)"
)
_PRIOR_SENTENCE = re.compile(r"[。！？!?；;.\n]")

_HOT = r"(?:高温|未冷却|热态|发烫|灼热|刚打印完|\b(?:hot|still[- ]hot|not cooled|before (?:it|the nozzle) (?:has )?cools? down|before (?:it|the nozzle) has cooled down)\b)"
_PART = r"(?:喷嘴|喷头|热床|加热块|\b(?:nozzle|print[- ]?head|hotend|heat(?:ed|er)? bed|heater block)\b)"
_TOUCH = r"(?:触摸|触碰|碰触|摸|探一下|\b(?:touch(?:ing)?|feel(?:ing)?|make contact with)\b)"
_NOZZLE = r"(?:喷嘴|喷头|\b(?:nozzle|print[- ]?head|hotend)\b)"
_REMOVE = r"(?:拆卸|拆下|拧下|拆掉|拆除|拧下来|\b(?:remove|unscrew|take off|detach)\b)"
_LIVE = (
    r"(?:带电|不断电|未断电|不用断电|无需断电|(?:仍然|仍)?通电(?!前)|上电时"
    r"|\b(?:energized|energised|live|powered on|(?:mains |the )?power (?:is )?(?:still )?(?:on|connected)"
    r"|without (?:disconnecting|isolating|cutting|shutting off) (?:the )?power)\b)"
)
_ELECTRICAL_PART = (
    r"(?:加热器|加热棒|热敏电阻|端子|线束|接线|接头|插头|电箱|电气舱|控制板"
    r"|\b(?:heater|thermistor|power supply|wires?|wiring|leads?|terminals?|connectors?|plugs?"
    r"|(?:electronics|electrical) (?:enclosure|compartment)|control board)\b)"
)
_ELECTRICAL_WORK = (
    r"(?:拆卸|拆开|打开|拆下|拧下|拔出|拔下|拔掉|拔除|插拔|测量"
    r"|\b(?:disconnect|unplug|unclip|remove|detach|pull (?:out|off)|rewire|probe|open)\b)"
)
_HAND_MOTION = (
    r"(?:用手|徒手|手动).{0,12}?(?:移动|推|推动|扳动|转动).{0,12}?(?:电机|丝杆|皮带|滑台|轴)"
    r"|\b(?:push|move|turn|rotate)\b.{0,38}?\b(?:carriage|motor|axis|belt|lead screw)\b"
    r".{0,22}?\b(?:by hand|manually|with your hands?)\b"
    r"|\b(?:manually|by hand)\b.{0,18}?\b(?:push|move|turn|rotate)\b"
    r".{0,32}?\b(?:carriage|motor|axis|belt|lead screw)\b"
)
_BYPASS = (
    r"(?:绕过|跳过|关闭|禁用|短接|屏蔽|忽略|无视"
    r"|\b(?:bypass(?:ing)?|disable|disabling|deactivate|override|ignore|disregard|bridge|short(?: out)?|turn off|switch off)\b)"
)
_PROTECTION = (
    r"(?:温度保护|热保护|过热保护|热失控保护|安全保护|保护开关|看门狗|限位(?:保护|开关)|行程(?:保护|开关)"
    r"|\b(?:thermal (?:runaway )?(?:protection|cut[- ]?off|safety|shutdown)|overheat(?:ing)? protection"
    r"|end[- ]?stop(?: protection)?|limit[- ]switch(?: protection)?|watchdog|safety (?:interlock|switch|cut[- ]?off))\b)"
)

def _near(first: str, second: str, distance: int) -> str:
    return rf"(?:{first}).{{0,{distance}}}?(?:{second})"

def _in_either_order(first: str, second: str, distance: int) -> str:
    return rf"(?:{_near(first, second, distance)}|{_near(second, first, distance)})"

_HOT_PART = _in_either_order(_HOT, _PART, 34)
_NOZZLE_REMOVAL = (
    _near(_REMOVE, _NOZZLE, 22) + "|" + _near(r"(?:把|将)" + _NOZZLE, _REMOVE, 12)
)
_ELECTRICAL_SERVICE = _in_either_order(_ELECTRICAL_WORK, _ELECTRICAL_PART, 34)
_NOZZLE_MENTION = re.compile(_NOZZLE)
_PROTECTION_MENTION = re.compile(_PROTECTION)
_REMOVE_ACTION = re.compile(_REMOVE)
_ELECTRICAL_ACTION = re.compile(_ELECTRICAL_WORK)
_HOT_PRONOUN_REMOVAL = re.compile(_in_either_order(_HOT, _REMOVE + r" it\b", 40))
_PRONOUN_BYPASS = re.compile(r"(?:将其关闭|把它禁用|\b(?:turn it off|switch it off|disable it|bypass it|ignore it)\b)")
_OPERATE = r"(?:继续(?:加热|运行|打印|移动)|\b(?:continue|keep|resume) (?:heating|printing|operating|moving)\b)"
_PROTECTION_OFF = _near(_PROTECTION, r"(?:已关闭|被关闭|已禁用|\b(?:disabled|off|bypassed)\b)", 20)
_BYPASS_TARGET = (
    _BYPASS + r"(?:(?!电源|供电|再(?:检查|查看|核对)|\b(?:mains|power|before|after|then)\b).){0,32}?" + _PROTECTION
)
_PATTERNS = (
    ("TOUCH_HOT_PART", re.compile(
        _near(_TOUCH, _HOT_PART, 40)
        + r"|(?:手背|手指|指尖|用手|徒手).{0,16}?(?:触摸|触碰|摸|对比|感受|试温).{0,16}?(?:喷嘴|喷头|热床|加热|高温|灼热)"
    ), re.compile(_TOUCH + r"|对比|感受|试温")),
    ("DISASSEMBLE_HOT", re.compile(
        _in_either_order(_HOT, _NOZZLE_REMOVAL, 40) + "|" + _near(_near(_REMOVE, _HOT, 18), _NOZZLE, 18)
        + "|" + _near(_HOT_PART, _REMOVE + r" (?:it|the nozzle)\b", 28)
    ), _REMOVE_ACTION),
    ("LIVE_ELECTRICAL", re.compile(
        _in_either_order(_LIVE, _ELECTRICAL_SERVICE, 65)
        + "|" + _near(_near(_ELECTRICAL_WORK, _LIVE, 26), _ELECTRICAL_PART, 26)
        + "|" + _near(_near(_ELECTRICAL_PART, _LIVE, 26), _ELECTRICAL_WORK, 26)
        + r"|(?:带电|不断电|未断电|通电状态下).{0,8}?(?:拆卸|拆开|插拔)"
    ), _ELECTRICAL_ACTION),
    ("BYPASS_PROTECTION", re.compile(
        _BYPASS_TARGET
        + rf"|(?:把|将|\bset\b).{{0,14}}?{_PROTECTION}.{{0,16}}?(?:设为不参与|设为关闭|置为关闭|\bto off\b)"
        + rf"|\b(?:turn|switch)\b.{{0,14}}?{_PROTECTION}.{{0,12}}?\boff\b"
        + "|" + _in_either_order(_OPERATE, _PROTECTION_OFF, 40)
    ), re.compile(_BYPASS + r"|设为|置为|\b(?:set|turn|switch)\b|" + _OPERATE)),
    ("DANGEROUS_MOTION", re.compile(_in_either_order(_LIVE, _HAND_MOTION, 45)),
     re.compile(r"移动|推动|推|扳动|转动|\b(?:push|move|turn|rotate)\b")),
    ("DEFEAT_SAFETY_TEST", re.compile(
        r"(?:关闭|禁用|绕过).{0,10}?(?:安全|保护).{0,10}?(?:测试|验证|运行|打印)"
    ), re.compile(_BYPASS)),
)
_POWER_ISOLATION = re.compile(
    r"先断电|断电后|断电降温后|断电并|断电且|确认.{0,8}?断电|拔掉电源|切断电源"
    r"|\b(?:after unplugging (?:the |your )?(?:printer|machine|unit)"
    r"|after (?:disconnecting|isolating) (?:the )?power|after (?:the )?(?:machine|printer|unit) is off"
    r"|with (?:the )?power (?:off|disconnected)|power (?:off|down) first)\b"
)
_COOLING = re.compile(r"(?:完全)?冷却后|冷却至|断电降温后|\b(?:has cooled|cooled completely|fully cooled|is cold)\b")
_UNKNOWN_POWER = re.compile(
    r"未确认|无法确认|不确定|\b(?:unknown|uncertain|not sure|do not know|don't know|unverified)\b"
)


@dataclass(frozen=True)
class SafetyFinding:
    category: str
    field: str


@dataclass(frozen=True)
class SafetyReview:
    findings: tuple[SafetyFinding, ...]

    @property
    def triggered(self) -> bool:
        return bool(self.findings)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({finding.category for finding in self.findings}))

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(sorted({finding.field for finding in self.findings}))

    @property
    def status(self) -> str:
        if not self.triggered:
            return "clear"
        return "review_required" if self.categories == (_REVIEW_REQUIRED,) else "blocked"

    def as_dict(self) -> dict:
        return {"blocked": self.triggered, "status": self.status, "categories": list(self.categories)}

    def summary(self) -> str:
        if self.status == "review_required":
            return "SAFETY_REVIEW_REQUIRED：维修建议缺少可确认的安全前置条件，已隐藏，需人工复核。"
        return (f"SAFETY_BLOCKED：模型建议触发安全审查（{','.join(self.categories)}），"
                "高风险内容未进入报告。")


def _remove_prohibited_examples(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        before = _PRIOR_SENTENCE.split(text[:match.start()])[-1]
        after = _PRIOR_SENTENCE.split(text[match.end():], maxsplit=1)[0]
        if _REPORTS.search(before) and _PROHIBITED_EXAMPLE.search(after):
            return " "
        return match.group()
    return _QUOTE.sub(replace, text)


def _is_prohibited(clause: str, match: re.Match[str], action: re.Pattern[str]) -> bool:
    # A live/hot state may precede a prohibition. Find the action inside the
    # risk span so the condition itself does not determine the negation scope.
    operation = action.search(clause, match.start(), match.end())
    prefix = clause[:operation.start() if operation else match.start()]
    suffix = clause[match.end():]
    if _RISK_VERDICT.match(suffix) or (_DISCUSSION.match(prefix) and _RISK_DESCRIPTION.search(suffix)):
        return True
    cues = list(_FORBID.finditer(prefix))
    if not cues or _EXCEPTION.search(suffix):
        return False
    scope = prefix[cues[-1].end():]
    return not (_FALSE_NEGATION.match(scope) or _PERMISSION.match(scope))


def _service_needs_review(clause: str) -> bool:
    power_isolated = bool(_POWER_ISOLATION.search(clause)) and not _UNKNOWN_POWER.search(clause)
    nozzle_ready = power_isolated and bool(_COOLING.search(clause))
    if not power_isolated and any(not _is_prohibited(clause, match, _ELECTRICAL_ACTION)
                                  for match in re.finditer(_ELECTRICAL_SERVICE, clause)):
        return True
    return not nozzle_ready and any(not _is_prohibited(clause, match, _REMOVE_ACTION)
                                    for match in re.finditer(_NOZZLE_REMOVAL, clause))


def _clauses(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return [clause for clause in _SCOPE_BREAK.split(_remove_prohibited_examples(normalized)) if clause.strip()]


def _review_clauses(clauses: list[str], previous: str = "") -> tuple[str, ...]:
    categories = set()
    for clause in clauses:
        hits = {category for category, pattern, action in _PATTERNS
                if any(not _is_prohibited(clause, match, action) for match in pattern.finditer(clause))}
        if _NOZZLE_MENTION.search(previous) and any(not _is_prohibited(clause, match, _REMOVE_ACTION)
                                                   for match in _HOT_PRONOUN_REMOVAL.finditer(clause)):
            hits.add("DISASSEMBLE_HOT")
        if _PROTECTION_MENTION.search(previous) and any(not _is_prohibited(clause, match, _PRONOUN_BYPASS)
                                                       for match in _PRONOUN_BYPASS.finditer(clause)):
            hits.add("BYPASS_PROTECTION")
        if not hits and _service_needs_review(clause):
            hits.add(_REVIEW_REQUIRED)
        categories.update(hits)
        previous = clause
    return tuple(sorted(categories))


def review_text(text: str) -> tuple[str, ...]:
    """Return risks in the matched scopes. Unverified service is withheld for review."""
    return _review_clauses(_clauses(text)) if text else ()


def review_fields(fields: Iterable[tuple[str, str]]) -> SafetyReview:
    return SafetyReview(tuple(SafetyFinding(category, name) for name, text in fields
                              for category in review_text(text)))


def review_draft(draft: SFTDiagnosisDraft) -> SafetyReview:
    fields = [(key, getattr(draft, key)) for key in (
        "fault_category", "suspected_cause", "supporting_observation", "safety_note"
    )]
    findings = list(review_fields(fields).findings)
    previous = ""
    for i, check in enumerate(draft.recommended_checks):
        clauses = _clauses(check)
        findings.extend(SafetyFinding(category, f"recommended_checks[{i}]")
                        for category in _review_clauses(clauses, previous))
        if clauses:
            previous = clauses[-1]
    return SafetyReview(tuple(findings))


def public_safety(content: str) -> dict:
    try:
        draft = validate_diagnosis(content)
    except DiagnosisValidationError:
        return {"blocked": None, "status": "not_evaluated", "categories": []}
    return review_draft(draft).as_dict()


def safe_checks(draft: SFTDiagnosisDraft, review: SafetyReview) -> Sequence[str]:
    """Drop only the checks that themselves triggered a safety finding."""
    risky = {finding.field for finding in review.findings
             if finding.field.startswith("recommended_checks[")}
    return [check for index, check in enumerate(draft.recommended_checks)
            if f"recommended_checks[{index}]" not in risky]
