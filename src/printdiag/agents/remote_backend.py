"""Windows-side client for the authenticated cloud inference service.

Only a loopback origin is accepted (the SSH tunnel endpoint). Every call is
token authenticated, schema-validated and bounded. ``live_inference`` becomes
true only after a real successful ``/v1/generate`` response that matches the
requested mode and the locally audited adapter SHA-256.

Error codes follow the P10-5C contract:
REMOTE_UNAVAILABLE, REMOTE_TIMEOUT, REMOTE_AUTH_FAILED, REMOTE_INVALID_RESPONSE,
MODEL_NOT_READY, MODEL_GENERATION_FAILED (plus REMOTE_WRONG_SERVICE /
REMOTE_ENDPOINT_MISSING for deployment diagnostics).
"""

from collections.abc import Sequence
import os
import time
from uuid import uuid4

import httpx

from printdiag.agents.inference_backend import (
    BackendMode, BackendUnavailable, GenerationConfig, GenerationResult, ModelInfo,
)

MAX_RESPONSE_BYTES = 256 * 1024
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
EXPECTED_SERVICE = "printdiag-cloud-inference"


class RemoteBackendError(RuntimeError):
    """The cloud inference call failed; ``code`` is a stable error identifier."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class RemoteInferenceBackend:
    """InferenceBackend implementation for the authenticated cloud service."""

    name = "remote-cloud"
    live_inference_capable = True

    def __init__(self, base_url: str, token: str, *, model: BackendMode = "lora",
                 timeout_s: float = 180.0, expected_adapter_sha256: str | None = None,
                 client: httpx.Client | None = None,
                 expected_service: str = EXPECTED_SERVICE) -> None:
        if model not in ("base", "lora"):
            raise ValueError("model must be 'base' or 'lora'")
        try:
            url = httpx.URL(base_url)
        except (ValueError, httpx.InvalidURL):
            raise ValueError("remote backend URL is invalid") from None
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
            raise ValueError("remote backend URL must be an HTTP(S) loopback origin")
        if url.host.casefold() not in LOOPBACK_HOSTS:
            raise ValueError("remote inference must go through a loopback tunnel endpoint")
        if len(token) < 16:
            raise ValueError("remote token must be at least 16 characters")
        if not 0 < timeout_s <= 600:
            raise ValueError("timeout_s must be in (0, 600]")
        port = url.port or (443 if url.scheme == "https" else 80)
        host = url.host
        if ":" in host:  # IPv6 literal
            host = f"[{host}]"
        self._origin = f"{url.scheme}://{host}:{port}"
        self._token = token
        self.mode = model
        self._timeout = timeout_s
        self._expected_adapter_sha256 = expected_adapter_sha256
        self._expected_service = expected_service
        self._client = client
        self._loaded = False
        self._info: ModelInfo | None = None
        self._blockers: tuple[str, ...] = ()

    # ---- http -----------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs) -> dict:
        headers = {"X-PrintDiag-Token": self._token, "Accept": "application/json"}
        client = self._client or httpx.Client(
            trust_env=False, follow_redirects=False, timeout=self._timeout)
        try:
            response = client.request(method, self._origin + path, headers=headers, **kwargs)
        except httpx.TimeoutException:
            raise RemoteBackendError("REMOTE_TIMEOUT", "cloud inference request timed out") from None
        except httpx.HTTPError as exc:
            raise RemoteBackendError("REMOTE_UNAVAILABLE", type(exc).__name__) from None
        finally:
            if self._client is None:
                client.close()
        if response.status_code == 401:
            raise RemoteBackendError("REMOTE_AUTH_FAILED", "cloud rejected the token")
        if response.status_code == 404:
            raise RemoteBackendError(
                "REMOTE_ENDPOINT_MISSING",
                f"{path} not found; deploy the P10-5C cloud service")
        if response.status_code == 413:
            raise RemoteBackendError("REMOTE_INVALID_RESPONSE", "request body too large")
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise RemoteBackendError("REMOTE_INVALID_RESPONSE", "response exceeds size limit")
        if response.status_code == 503:
            raise RemoteBackendError("MODEL_NOT_READY", "cloud model is not ready")
        if response.status_code >= 500:
            raise RemoteBackendError("MODEL_GENERATION_FAILED",
                                     f"cloud generation failed (HTTP {response.status_code})")
        if response.status_code != 200:
            raise RemoteBackendError("REMOTE_INVALID_RESPONSE",
                                     f"unexpected HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            raise RemoteBackendError("REMOTE_INVALID_RESPONSE", "response is not JSON") from None
        if not isinstance(payload, dict):
            raise RemoteBackendError("REMOTE_INVALID_RESPONSE", "response is not an object")
        return payload

    # ---- InferenceBackend ----------------------------------------------
    def load(self) -> ModelInfo:
        started = time.perf_counter()
        try:
            health = self._request("GET", "/healthz")
        except RemoteBackendError as exc:
            self._blockers = (str(exc),)
            raise BackendUnavailable(self._blockers) from None
        blockers: list[str] = []
        service = health.get("service")
        if service != self._expected_service:
            blockers.append(
                f"wrong service on the tunnel: {service!r}; expected {self._expected_service!r} "
                "(the old printdiag-cloud-test smoke service must be stopped)")
        if health.get("model_ready") is not True:
            blockers.append("cloud reports model_ready=false")
        inventory: dict = {}
        try:
            inventory = self._request("GET", "/v1/model-info")
        except RemoteBackendError as exc:
            blockers.append(str(exc))
        adapter_sha = None
        if inventory:
            modes = inventory.get("available_modes") or []
            if self.mode not in modes:
                blockers.append(f"cloud does not expose the '{self.mode}' mode")
            adapter_sha = inventory.get("adapter_sha256")
            if health.get("adapter_ready") is not True and self.mode == "lora":
                blockers.append("cloud reports adapter_ready=false")
            if (self.mode == "lora" and self._expected_adapter_sha256
                    and adapter_sha != self._expected_adapter_sha256):
                blockers.append("cloud adapter SHA-256 does not match the local audited adapter")
        if blockers:
            self._blockers = tuple(blockers)
            raise BackendUnavailable(self._blockers)
        self._loaded = True
        self._info = ModelInfo(
            backend=self.name, mode=self.mode, model_path=self._origin, adapter_path=None,
            adapter_sha256=adapter_sha if self.mode == "lora" else None,
            dtype=inventory.get("inference_dtype"), device="remote-gpu",
            load_ms=round((time.perf_counter() - started) * 1000, 2),
            live_inference_capable=True,
            architecture=inventory.get("architecture"), gpu_peak_mb=None)
        return self._info

    def unload(self) -> None:
        self._loaded = False
        self._info = None

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_info(self) -> ModelInfo | None:
        return self._info

    def generate(self, messages: Sequence[dict[str, str]],
                 config: GenerationConfig) -> GenerationResult:
        if not self._loaded:
            raise BackendUnavailable(self._blockers or ("remote backend is not loaded",))
        request_id = uuid4().hex
        started = time.perf_counter()
        payload = self._request("POST", "/v1/generate", json={
            "request_id": request_id,
            "mode": self.mode,
            "messages": [{"role": item["role"], "content": item["content"]} for item in messages],
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "temperature": config.temperature if config.do_sample else None,
            "inference_options": {
                "top_p": config.top_p, "seed": config.seed,
                "enable_thinking": config.enable_thinking,
            },
        })
        text = payload.get("raw_output")
        if payload.get("live_inference") is not True:
            raise RemoteBackendError("REMOTE_INVALID_RESPONSE",
                                     "cloud did not claim live inference")
        if payload.get("mode") != self.mode:
            raise RemoteBackendError(
                "REMOTE_INVALID_RESPONSE",
                f"cloud returned mode {payload.get('mode')!r}, expected {self.mode!r}")
        metadata = payload.get("model_metadata") or {}
        if self.mode == "lora" and self._expected_adapter_sha256 and metadata.get(
                "adapter_sha256") not in (None, self._expected_adapter_sha256):
            raise RemoteBackendError("REMOTE_INVALID_RESPONSE", "cloud adapter SHA mismatch")
        if not isinstance(text, str) or not text.strip():
            raise RemoteBackendError("MODEL_GENERATION_FAILED", "cloud returned empty text")
        wall_ms = round((time.perf_counter() - started) * 1000, 2)
        return GenerationResult(
            text=text, backend=self.name, mode=self.mode, live_inference=True,
            latency_ms=float(payload.get("inference_latency_ms") or wall_ms),
            gpu_peak_mb=metadata.get("gpu_peak_mb"), prompt_chars=sum(
                len(item["content"]) for item in messages),
            input_tokens=payload.get("input_tokens"), output_tokens=payload.get("output_tokens"),
            request_id=payload.get("request_id") or request_id,
            model_metadata=metadata)

    @classmethod
    def from_environment(cls, *, expected_adapter_sha256: str | None = None) -> "RemoteInferenceBackend":
        if os.environ.get("PRINTDIAG_INFERENCE_BACKEND", "").strip().lower() not in {"", "remote"}:
            raise BackendUnavailable(("PRINTDIAG_INFERENCE_BACKEND must be 'remote'",))
        if os.environ.get("PRINTDIAG_ALLOW_REMOTE_INFERENCE") != "1":
            raise BackendUnavailable((
                "remote inference is not authorized; set PRINTDIAG_ALLOW_REMOTE_INFERENCE=1",))
        base_url = (os.environ.get("PRINTDIAG_REMOTE_INFERENCE_URL")
                    or os.environ.get("PRINTDIAG_CLOUD_URL")
                    or "http://127.0.0.1:18008")
        token = (os.environ.get("PRINTDIAG_REMOTE_INFERENCE_TOKEN")
                 or os.environ.get("PRINTDIAG_CLOUD_TOKEN") or "")
        model = os.environ.get("PRINTDIAG_REMOTE_MODEL", "lora")
        if not token:
            raise BackendUnavailable(("PRINTDIAG_REMOTE_INFERENCE_TOKEN is not configured",))
        if expected_adapter_sha256 is None:
            try:
                from printdiag.agents.local_lora import inspect_artifacts

                expected_adapter_sha256 = inspect_artifacts().adapter_sha256
            except Exception:  # noqa: BLE001 - keep None when local artifacts are unavailable
                expected_adapter_sha256 = None
        return cls(base_url, token, model=model,
                   expected_adapter_sha256=expected_adapter_sha256)
