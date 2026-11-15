"""Bounded request-local orchestration of the Provider, tools and report."""

import asyncio
from collections.abc import Callable
import logging
from math import isfinite
from threading import Event
from time import monotonic
from uuid import uuid4

from printdiag.agents.providers import Exchange, Provider, ProviderFailure, SFTDraftTurn
from printdiag.reporting.agent_report import ReportDraft, build_report
from printdiag.schemas.agent import DiagnosisReport, DiagnosisRequest, ToolCallRequest, ToolResult
from printdiag.tools.registry import ToolRegistry, ToolRejected

_LOG = logging.getLogger(__name__)


def _provider_notes(provider: Provider, results: "list[ToolResult]") -> tuple[str, ...]:
    """Optional provider-supplied limitations (rule routing, provenance labels)."""
    notes_for = getattr(provider, "notes_for", None)
    if callable(notes_for):
        try:
            return tuple(str(note) for note in notes_for(results))
        except Exception:  # noqa: BLE001 - notes must never break a diagnosis
            return ()
    return tuple(str(note) for note in getattr(provider, "notes", ()))


class DiagnosisWorkflow:
    def __init__(self, provider: Provider | Callable[[str], Provider], registry: ToolRegistry, *,
                 max_tool_calls: int = 6, total_s: float = 12.0, tool_s: float = 2.5,
                 draft_adapter: Callable[..., DiagnosisReport] | None = None) -> None:
        if (type(max_tool_calls) is not int or not 1 <= max_tool_calls <= 6
                or not all(isfinite(value) and value > 0 for value in (total_s, tool_s)) or tool_s > total_s):
            raise ValueError("invalid diagnosis budgets")
        self._provider = provider
        self._registry = registry
        self._max_tool_calls = max_tool_calls
        self._total_s = total_s
        self._tool_s = tool_s
        self._draft_adapter = draft_adapter

    @property
    def knowledge_available(self) -> bool:
        return self._registry.knowledge_available

    async def diagnose(self, request: DiagnosisRequest) -> DiagnosisReport:
        diagnosis_id = uuid4().hex
        provider = self._provider(diagnosis_id) if callable(self._provider) else self._provider
        deadline = monotonic() + self._total_s
        history: list[Exchange] = []
        results: list[ToolResult] = []
        seen_calls = set()
        seen_ids = set()
        draft = None
        sft_turn = None
        failure = None
        try:
            async with asyncio.timeout(self._total_s):
                while monotonic() < deadline:
                    turn = await provider.next_turn(request, self._registry.specifications(), tuple(history))
                    if monotonic() >= deadline:
                        raise TimeoutError
                    if isinstance(turn, ReportDraft):
                        draft = turn
                        break
                    if isinstance(turn, SFTDraftTurn):
                        sft_turn = turn
                        break
                    if not isinstance(turn, list) or not turn or not all(
                        isinstance(call, ToolCallRequest) for call in turn
                    ):
                        raise ProviderFailure("INVALID_PROVIDER_RESPONSE")
                    if len(turn) + len(results) > self._max_tool_calls:
                        raise ProviderFailure("BUDGET_EXHAUSTED")
                    prepared = [self._registry.prepare(call) for call in turn]
                    ids = {call.call_id for call in prepared}
                    fingerprints = {call.fingerprint() for call in prepared}
                    if len(ids) != len(prepared) or ids & seen_ids:
                        raise ProviderFailure("INVALID_PROVIDER_RESPONSE")
                    if len(fingerprints) != len(prepared) or fingerprints & seen_calls:
                        raise ProviderFailure("BUDGET_EXHAUSTED")
                    seen_ids.update(ids)
                    seen_calls.update(fingerprints)
                    batch = []
                    for call in prepared:
                        started = monotonic()
                        budget = min(self._tool_s, deadline - started)
                        if budget <= 0:
                            raise TimeoutError
                        cancelled = Event()
                        try:
                            async with asyncio.timeout(budget):
                                result = await asyncio.to_thread(
                                    self._registry.execute, call, started + budget, cancelled=cancelled,
                                )
                            if monotonic() >= started + budget:
                                raise TimeoutError
                        except TimeoutError:
                            result = self._registry.error(call.tool, "BUDGET_EXHAUSTED")
                        finally:
                            cancelled.set()
                        results.append(result)
                        batch.append(result)
                        _LOG.info("diagnosis=%s tool=%s status=%s evidence=%s issues=%s elapsed_ms=%.2f",
                                  diagnosis_id, result.tool, result.status, result.evidence_id,
                                  ",".join(issue.code for issue in result.issues), (monotonic() - started) * 1000)
                    history.append(Exchange(tuple(turn), tuple(batch)))
                else:
                    raise TimeoutError
        except (ToolRejected, ProviderFailure) as exc:
            failure = exc.code
        except TimeoutError:
            failure = "BUDGET_EXHAUSTED"

        if sft_turn is not None and self._draft_adapter is not None and failure is None:
            report = self._draft_adapter(
                diagnosis_id, sft_turn.content, results=results, user_observation=request.symptom,
                mock=provider.mock, extra_limitations=_provider_notes(provider, results),
            )
        else:
            if sft_turn is not None and failure is None:
                failure = "INVALID_PROVIDER_RESPONSE"
            report = build_report(diagnosis_id, mock=provider.mock, results=results, draft=draft,
                                  failure=failure, extra_limitations=_provider_notes(provider, results))
        _LOG.info("diagnosis=%s status=%s issues=%s", diagnosis_id, report.status,
                  ",".join(sorted({item.code for item in report.limitations if item.code is not None})))
        return report
