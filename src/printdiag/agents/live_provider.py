"""Opt-in configuration and request-local sessions for the live LLM path."""

from dataclasses import dataclass, field
import os
import re

import httpx

from printdiag.agents.providers import CompletionLimits, OpenAICompatibleProvider, TokenField, completion_endpoint


class LiveProviderPending(ValueError):
    """Live provider settings are missing, invalid or disabled."""


def _integer(name: str, default: str, ceiling: int) -> int:
    try:
        value = int(os.environ.get(name, default))
        if 1 <= value <= ceiling:
            return value
    except ValueError:
        pass
    raise LiveProviderPending(f"live_provider_pending: invalid {name}")


@dataclass(frozen=True)
class LiveProviderSettings:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_s: float
    max_tokens: int
    max_rounds: int
    token_field: TokenField

    @classmethod
    def from_environment(cls) -> "LiveProviderSettings":
        if os.environ.get("PRINTDIAG_ENABLE_LIVE_LLM") != "1":
            raise LiveProviderPending("live_provider_pending: live LLM is disabled")
        base_url = os.environ.get("PRINTDIAG_LLM_BASE_URL", "")
        model = os.environ.get("PRINTDIAG_LLM_MODEL", "")
        key = os.environ.get("PRINTDIAG_LLM_API_KEY", "")
        if not re.fullmatch(r"[A-Za-z0-9._~+/=-]{8,4096}", key):
            raise LiveProviderPending("live_provider_pending: missing or invalid PRINTDIAG_LLM_API_KEY")
        if key in model or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}", model):
            raise LiveProviderPending("live_provider_pending: missing or invalid PRINTDIAG_LLM_MODEL")
        try:
            completion_endpoint(base_url, key)
        except ValueError:
            raise LiveProviderPending("live_provider_pending: invalid PRINTDIAG_LLM_BASE_URL") from None

        try:
            timeout = float(os.environ.get("PRINTDIAG_LLM_TIMEOUT", "5"))
        except ValueError:
            timeout = 0.0
        if not 0 < timeout <= 30:
            raise LiveProviderPending("live_provider_pending: invalid PRINTDIAG_LLM_TIMEOUT")

        token_field = os.environ.get("PRINTDIAG_LLM_TOKEN_PARAMETER", "max_completion_tokens")
        if token_field not in {"max_completion_tokens", "max_tokens"}:
            raise LiveProviderPending("live_provider_pending: invalid PRINTDIAG_LLM_TOKEN_PARAMETER")
        return cls(base_url, model, key, timeout,
                   _integer("PRINTDIAG_LLM_MAX_TOKENS", "1024", 8192),
                   _integer("PRINTDIAG_LLM_MAX_ROUNDS", "4", 7), token_field)

    def for_diagnosis(self, diagnosis_id: str, *, transport: httpx.AsyncBaseTransport | None = None
                      ) -> OpenAICompatibleProvider:
        limits = CompletionLimits(diagnosis_id, self.max_tokens, self.max_rounds, self.token_field)
        return OpenAICompatibleProvider(base_url=self.base_url, model=self.model, api_key=self.api_key,
                                        timeout_s=self.timeout_s, transport=transport, limits=limits)
