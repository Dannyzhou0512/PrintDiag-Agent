"""Validate arguments before dispatching one of the three approved tools."""

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from threading import Event
from uuid import uuid4

from pydantic import ValidationError

from printdiag.rag.maintenance import MaintenanceIndex
from printdiag.schemas.agent import (
    EndstopArguments, ErrorCode, Issue, KnowledgeArguments, KnowledgeData, PrinterStateArguments,
    Source, ToolCallRequest, ToolName, ToolResult,
)
from printdiag.tools.device_reads import DeviceReads

ToolArguments = PrinterStateArguments | EndstopArguments | KnowledgeArguments
_TOOLS: dict[ToolName, tuple[type[ToolArguments], str]] = {
    "read_printer_state": (PrinterStateArguments, "Read projected printer position, queue and health."),
    "read_endstop_components": (EndstopArguments, "Read effective and geometric endstop observations by axis."),
    "search_maintenance_knowledge": (KnowledgeArguments, "Search reviewed local maintenance fragments."),
}


class ToolRejected(Exception):
    def __init__(self, code: ErrorCode) -> None:
        self.code = code


@dataclass(frozen=True)
class PreparedCall:
    call_id: str
    tool: ToolName
    arguments: ToolArguments

    def fingerprint(self) -> tuple[str, str]:
        arguments = self.arguments.model_dump()
        if isinstance(self.arguments, EndstopArguments):
            arguments["axes"] = sorted(self.arguments.axes)
        if isinstance(self.arguments, KnowledgeArguments):
            arguments["query"] = " ".join(self.arguments.query.casefold().split())
        return self.tool, json.dumps(arguments, sort_keys=True)


class ToolRegistry:
    def __init__(self, device: DeviceReads, knowledge: MaintenanceIndex) -> None:
        self._device = device
        self._knowledge = knowledge

    @property
    def knowledge_available(self) -> bool:
        return self._knowledge.available

    @staticmethod
    def specifications() -> list[dict]:
        return [{"type": "function", "function": {
            "name": name, "description": description, "parameters": schema.model_json_schema(),
        }} for name, (schema, description) in _TOOLS.items()]

    @staticmethod
    def prepare(call: ToolCallRequest) -> PreparedCall:
        if call.tool not in _TOOLS:
            raise ToolRejected("TOOL_NOT_ALLOWED")
        schema, _ = _TOOLS[call.tool]
        try:
            arguments = schema.model_validate(call.arguments)
        except ValidationError:
            raise ToolRejected("INVALID_ARGUMENT") from None
        return PreparedCall(call.call_id, call.tool, arguments)

    def source_for(self, tool: ToolName) -> Source:
        return "curated_knowledge" if tool == "search_maintenance_knowledge" else self._device.source

    def error(self, tool: ToolName, code: ErrorCode) -> ToolResult:
        return ToolResult(tool=tool, status="error", source=self.source_for(tool), data=None,
                          evidence_id=None, observed_at_utc=None, issues=[Issue(code=code)])

    def execute(self, call: PreparedCall, deadline: float, *, cancelled: Event | None = None) -> ToolResult:
        arguments = call.arguments
        match arguments:
            case PrinterStateArguments():
                data, issues = self._device.printer_state(deadline, cancelled=cancelled)
                usable = data is not None
            case EndstopArguments():
                data, issues = self._device.endstops(arguments.axes, deadline, cancelled=cancelled)
                usable = any(value is not None for row in data.components
                             for value in (row.triggered, row.geometric_triggered, row.enabled))
            case KnowledgeArguments():
                if not self._knowledge.available:
                    return self.error(call.tool, "KNOWLEDGE_UNAVAILABLE")
                category = getattr(arguments, "category", None)
                if category is not None and hasattr(self._knowledge, "search_with_category"):
                    hits = self._knowledge.search_with_category(arguments.query, arguments.limit, category)
                else:
                    hits = self._knowledge.search(arguments.query, arguments.limit)
                data = KnowledgeData(hits=hits)
                issues = []
                usable = True
        provenance = None if call.tool == "search_maintenance_knowledge" else self._device.provenance
        return ToolResult(tool=call.tool, source=self.source_for(call.tool), data=data, issues=issues,
                          status=("partial" if issues else "ok") if usable else "error",
                          evidence_id=uuid4().hex if usable else None,
                          observed_at_utc=datetime.now(UTC) if usable else None,
                          provenance=provenance if usable else None)
