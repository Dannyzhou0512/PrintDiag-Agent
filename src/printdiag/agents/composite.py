"""Compose the deterministic read-only tool router with a draft generator.

The tool/evidence half stays rule-based (no LLM planning claim); only the final
seven-field draft comes from the injected generator, which can be the local LoRA
provider (when available) or an archived replay for offline comparison. The
draft still goes through ``validate_diagnosis`` → ``adapt_sft_report``.
"""

from collections.abc import Awaitable, Callable, Sequence
import inspect

from printdiag.agents.providers import Exchange, ProviderFailure, SFTDraftTurn
from printdiag.agents.symptom_routing import RuleBasedProvider
from printdiag.schemas.agent import DiagnosisRequest, ToolCallRequest, ToolResult

DrafterResult = str | None | Awaitable[str | None]
Drafter = Callable[[DiagnosisRequest, Sequence[Exchange]], DrafterResult]


class EvidenceGroundedDraftProvider:
    """First two turns: rule-based read-only tools; final turn: injected draft."""

    mock = False

    def __init__(self, router: RuleBasedProvider, drafter: Drafter, *, label: str) -> None:
        self._router = router
        self._drafter = drafter
        self.label = label

    async def next_turn(self, request: DiagnosisRequest, tools: list[dict],
                        history: Sequence[Exchange]) -> list[ToolCallRequest] | SFTDraftTurn:
        if len(history) <= 1:
            return await self._router.next_turn(request, tools, history)
        content = self._drafter(request, history)
        if inspect.isawaitable(content):
            content = await content
        if not isinstance(content, str) or not content.strip():
            raise ProviderFailure("INVALID_PROVIDER_RESPONSE")
        return SFTDraftTurn(content)

    def notes_for(self, results: Sequence[ToolResult]) -> tuple[str, ...]:
        return (*self._router.notes_for(results), f"草稿来源：{self.label}")
