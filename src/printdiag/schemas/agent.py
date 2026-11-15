"""Strict public contracts for the V0.1 diagnosis API and its three tools."""

from datetime import timedelta
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

Axis = Literal["x_min", "y_min", "z_min"]
ToolName = Literal["read_printer_state", "read_endstop_components", "search_maintenance_knowledge"]
FaultTopic = Literal[
    "喷嘴温度异常", "热床温度异常", "温度传感器异常", "加热器故障", "限位开关异常", "运动队列异常",
    "电机与运动系统异常", "欠挤出", "喷嘴堵塞", "打印件翘边", "首层附着异常",
    "设备无法启动或打印中断", "证据不足或正常状态",
]
Source = Literal["live_read", "mock", "curated_knowledge"]
ErrorCode = Literal[
    "TOOL_NOT_ALLOWED", "INVALID_ARGUMENT", "ACCESS_DENIED", "DEVICE_UNAVAILABLE", "DEVICE_TIMEOUT",
    "DEVICE_HTTP_ERROR", "INVALID_DEVICE_RESPONSE", "FIELD_UNAVAILABLE", "STALE_OBSERVATION",
    "KNOWLEDGE_UNAVAILABLE", "PROVIDER_UNAVAILABLE", "INVALID_PROVIDER_RESPONSE", "BUDGET_EXHAUSTED",
]
Finite = Annotated[float, Field(allow_inf_nan=False)]
NonnegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0, le=2**63 - 1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_-]{1,80}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$")]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class PrinterStateArguments(Contract):
    pass


class EndstopArguments(Contract):
    axes: Annotated[list[Axis], Field(min_length=1, max_length=3, json_schema_extra={"uniqueItems": True})]

    @model_validator(mode="after")
    def distinct_axes(self) -> Self:
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("axes must be unique")
        return self


class KnowledgeArguments(Contract):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
    limit: Annotated[int, Field(ge=1, le=5)] = 3
    # Optional category routing for the structured BM25 index; ignored by the
    # legacy v0.1 index and therefore backward compatible.
    category: FaultTopic | None = None


class XYZ(Contract):
    x: Finite
    y: Finite
    z: Finite


class PrinterStateProjection(Contract):
    sim_seconds: NonnegativeFinite | None
    position_mm: XYZ | None
    stepper_position: XYZ | None
    idle: bool | None
    moves_queued: Count | None
    timers_active: bool | None
    watchdog_resets: Count | None
    thermal_fault_count: Count | None
    # Temperature observability (P10-2). Every field is optional and defaults to
    # None: this simulator build exposes temperature/ambient/pwm/element flags
    # but no target temperature, so a missing target is reported as unavailable
    # instead of being invented. Additive fields only; the v1.0 tool-result
    # contract and the report shapes stay valid for existing callers.
    hotend_temperature: Finite | None = None
    hotend_target_temperature: Finite | None = None
    hotend_ambient_temperature: Finite | None = None
    hotend_pwm_duty: Finite | None = None
    hotend_element_ok: bool | None = None
    hotend_sensor_mode: ShortText | None = None
    hotend_sensor_error: Finite | None = None
    bed_temperature: Finite | None = None
    bed_target_temperature: Finite | None = None
    bed_ambient_temperature: Finite | None = None
    bed_pwm_duty: Finite | None = None
    bed_element_ok: bool | None = None
    bed_sensor_mode: ShortText | None = None
    bed_sensor_error: Finite | None = None


class EndstopComponentProjection(Contract):
    axis: Axis
    triggered: bool | None
    geometric_triggered: bool | None
    enabled: bool | None


class EndstopData(Contract):
    components: Annotated[list[EndstopComponentProjection], Field(min_length=1, max_length=3)]

    @model_validator(mode="after")
    def distinct_axes(self) -> Self:
        if len({row.axis for row in self.components}) != len(self.components):
            raise ValueError("component axes must be unique")
        return self


class KnowledgeHit(Contract):
    document_id: Identifier
    title: ShortText
    anchor: Identifier
    revision: Revision
    excerpt: ShortText
    # P10-3 retrieval provenance (optional, additive): chunk identity, auditable
    # source reference and the sparse retrieval score. Old callers can ignore it.
    chunk_id: Identifier | None = None
    source_reference: ShortText | None = None
    retrieval_score: Finite | None = None


class KnowledgeData(Contract):
    hits: Annotated[list[KnowledgeHit], Field(max_length=5)]


class Issue(Contract):
    code: ErrorCode
    field: Annotated[str, StringConstraints(max_length=80)] | None = None
    retryable: bool = False


class ToolResult(Contract):
    schema_version: Literal["1.0"] = "1.0"
    tool: ToolName
    status: Literal["ok", "partial", "error"]
    evidence_id: Identifier | None
    observed_at_utc: AwareDatetime | None
    source: Source
    # Declared by the executor, never by a model: a live read is either a local
    # simulator observation or (future) a physical printer observation.
    provenance: Literal["mock", "simulator"] | None = None
    data: PrinterStateProjection | EndstopData | KnowledgeData | None
    issues: Annotated[list[Issue], Field(max_length=20)]

    @model_validator(mode="after")
    def coherent_result(self) -> Self:
        expected = {
            "read_printer_state": PrinterStateProjection,
            "read_endstop_components": EndstopData,
            "search_maintenance_knowledge": KnowledgeData,
        }[self.tool]
        if (self.source == "curated_knowledge") != (expected is KnowledgeData):
            raise ValueError("source does not match tool")
        if self.data is not None and not isinstance(self.data, expected):
            raise ValueError("data does not match tool")
        if self.status == "error":
            if self.evidence_id is not None or self.observed_at_utc is not None or not self.issues:
                raise ValueError("failed tools cannot publish evidence")
            if self.provenance is not None:
                raise ValueError("failed tools publish no provenance")
        elif (self.evidence_id is None or self.observed_at_utc is None or self.data is None
              or self.observed_at_utc.utcoffset() != timedelta(0) or bool(self.issues) != (self.status == "partial")):
            raise ValueError("inconsistent observation metadata")
        if expected is KnowledgeData:
            if self.provenance is not None:
                raise ValueError("knowledge results carry no device provenance")
        elif self.status != "error":
            if self.provenance is None:
                raise ValueError("device evidence must declare provenance")
            if self.provenance == "simulator" and self.source != "live_read":
                raise ValueError("simulator provenance requires a live read")
            if self.provenance == "mock" and self.source != "mock":
                raise ValueError("mock provenance requires a mock source")
        return self


class DiagnosisRequest(Contract):
    symptom: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class ToolCallRequest(Contract):
    call_id: Identifier
    tool: Identifier
    arguments: dict[str, object]


class EvidenceKey(Contract):
    evidence_id: Identifier
    hit_index: Annotated[int, Field(ge=0, lt=5)] | None = None

    def key(self) -> tuple[str, int | None]:
        return self.evidence_id, self.hit_index


class EvidenceReference(EvidenceKey):
    tool: ToolName
    source: Source
    observed_at_utc: AwareDatetime
    data: PrinterStateProjection | EndstopData | KnowledgeHit

    @model_validator(mode="after")
    def coherent_reference(self) -> Self:
        expected = {
            "read_printer_state": PrinterStateProjection,
            "read_endstop_components": EndstopData,
            "search_maintenance_knowledge": KnowledgeHit,
        }[self.tool]
        if (not isinstance(self.data, expected) or self.observed_at_utc.utcoffset() != timedelta(0)
                or (self.source == "curated_knowledge") != (expected is KnowledgeHit)
                or (self.hit_index is not None) != (expected is KnowledgeHit)):
            raise ValueError("inconsistent evidence reference")
        return self


class Hypothesis(Contract):
    statement: ShortText
    inference: ShortText
    evidence_refs: Annotated[list[EvidenceKey], Field(min_length=1, max_length=6)]


class Limitation(Contract):
    code: ErrorCode | None = None
    detail: ShortText


class DiagnosisReport(Contract):
    diagnosis_id: Identifier
    status: Literal["complete", "partial", "insufficient_evidence"]
    summary: ShortText
    hypotheses: Annotated[list[Hypothesis], Field(max_length=4)]
    evidence_refs: Annotated[list[EvidenceReference], Field(max_length=30)]
    limitations: Annotated[list[Limitation], Field(max_length=25)]
    recommended_checks: Annotated[list[ShortText], Field(max_length=4)]

    @model_validator(mode="after")
    def resolved_references(self) -> Self:
        keys = {reference.key() for reference in self.evidence_refs}
        if len(keys) != len(self.evidence_refs) or any(
            ref.key() not in keys for hypothesis in self.hypotheses for ref in hypothesis.evidence_refs
        ):
            raise ValueError("report contains duplicate or unresolved evidence")
        return self
