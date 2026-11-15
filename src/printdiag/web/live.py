"""Server-side Live Qwen inference for the web UI.

Token, budget and concurrency stay on the server. No request is sent at startup;
the default GPU budget is 0. V2 is passed per request, never via global env.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import asyncio
import os
import threading
import time
from uuid import uuid4

import httpx

from printdiag.agents.composite import EvidenceGroundedDraftProvider
from printdiag.agents.diagnosis_prompt import TrainingTemplate
from printdiag.agents.inference_backend import BackendUnavailable, FakeBackend, InferenceBackend
from printdiag.agents.local_lora import LocalLoRAProvider, inspect_artifacts
from printdiag.agents.remote_backend import RemoteBackendError, RemoteInferenceBackend
from printdiag.agents.symptom_routing import RuleBasedProvider
from printdiag.agents.workflow import DiagnosisWorkflow
from printdiag.rag.structured import packaged_structured_index
from printdiag.reliability.v2.config import V2Config
from printdiag.reliability.v2.output_guard import guard_output
from printdiag.reporting.lora_report import adapt_sft_report
from printdiag.reporting.safety_review import public_safety
from printdiag.schemas.agent import DiagnosisRequest
from printdiag.tools.device_reads import simulator_device
from printdiag.tools.registry import ToolRegistry
from printdiag.web.answers import live_answer
from printdiag.web.budget import AuditLogger, BudgetError, BudgetStore

DEFAULT_URL = "http://127.0.0.1:18008"


class LiveError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 503) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"{code}: {message}")


@dataclass
class _Counters:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    invalid_outputs: int = 0
    http_requests: int = 0
    http_successes: int = 0
    http_failures: int = 0


class _ObservedRegistry(ToolRegistry):
    """Emit real stage events around actual read-only tool execution."""

    def __init__(self, device, knowledge, observer) -> None:
        super().__init__(device, knowledge)
        self._observer = observer

    def execute(self, call, deadline, *, cancelled=None):
        is_knowledge = call.tool == "search_maintenance_knowledge"
        self._observer("retrieving_knowledge" if is_knowledge else "reading_device",
                       "正在检索维修知识…" if is_knowledge else "正在读取设备状态…")
        return super().execute(call, deadline, cancelled=cancelled)


class LiveInferenceService:
    def __init__(self, *, base_url: str | None = None, token: str | None = None,
                 budget: int | None = None, max_concurrency: int | None = None,
                 allow_remote: bool | None = None,
                 backend_factory: Callable[[str], InferenceBackend] | None = None,
                 device_factory=None, budget_store: BudgetStore | None = None,
                 audit_logger: AuditLogger | None = None,
                 template: TrainingTemplate | None = None) -> None:
        self._base_url = base_url or os.environ.get(
            "PRINTDIAG_REMOTE_INFERENCE_URL",
            os.environ.get("PRINTDIAG_CLOUD_URL", DEFAULT_URL))
        self._token = token if token is not None else (
            os.environ.get("PRINTDIAG_REMOTE_INFERENCE_TOKEN")
            or os.environ.get("PRINTDIAG_CLOUD_TOKEN") or "")
        self._allow_remote = (allow_remote if allow_remote is not None
                              else os.environ.get("PRINTDIAG_ALLOW_REMOTE_INFERENCE") == "1")
        self._budget = budget if budget is not None else _int_env("PRINTDIAG_LIVE_GPU_BUDGET", 0)
        state_path = os.environ.get("PRINTDIAG_LIVE_BUDGET_STATE")
        self._store = budget_store or (BudgetStore(state_path) if state_path else None)
        if self._store is not None:
            self._store.configure(self._budget)
        audit_dir = os.environ.get("PRINTDIAG_LIVE_AUDIT_DIR")
        self._audit = audit_logger or (AuditLogger(audit_dir) if audit_dir else None)
        concurrency = (max_concurrency if max_concurrency is not None
                       else _int_env("PRINTDIAG_LIVE_MAX_CONCURRENCY", 1))
        self._semaphore = threading.BoundedSemaphore(max(1, concurrency))
        self._lock = threading.Lock()
        self._counters = _Counters()
        self._backend_factory = backend_factory
        self._template = template
        self._device_factory = device_factory or simulator_device
        self._last_success_utc: str | None = None
        self._last_error: str | None = None

    # ---- status ---------------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self._allow_remote and self._token)

    def status(self) -> dict:
        with self._lock:
            if self._store is not None:
                store = self._store.status()
                budget = store["budget"]
                used = store["used"]
                remaining = store["remaining"]
            else:
                budget = self._budget
                used = self._counters.attempted
                remaining = max(0, budget - used)
            counters = self._counters.__dict__.copy()
        reason = None
        if not self._allow_remote:
            reason = "PRINTDIAG_ALLOW_REMOTE_INFERENCE is not 1"
        elif not self._token:
            reason = "PRINTDIAG_REMOTE_INFERENCE_TOKEN is not configured"
        elif self._budget <= 0:
            reason = "PRINTDIAG_LIVE_GPU_BUDGET is 0"
        return {
            "available": self.configured and remaining > 0,
            "configured": self.configured,
            "budget": budget,
            "used": used,
            "remaining": remaining,
            "models": ["base", "lora"],
            "cloud_configured": bool(self._base_url),
            "reason": reason,
            "auto_call": False,
            "live_inference": True,
            "mock": False,
            "last_success_utc": self._last_success_utc,
            "last_error": self._last_error,
            **counters,
        }

    # ---- live diagnose --------------------------------------------------
    def diagnose(self, *, symptom: str, model: str, v2: bool,
                 request_id: str | None = None, stage_callback=None) -> dict:
        emit = stage_callback or (lambda key, label: None)
        if model not in ("base", "lora"):
            raise LiveError("INVALID_MODEL", "model must be base or lora", 422)
        if not self.configured:
            raise LiveError("LIVE_UNAVAILABLE", self.status()["reason"] or "live inference is not configured")
        if self._budget <= 0:
            raise LiveError("LIVE_UNAVAILABLE", "PRINTDIAG_LIVE_GPU_BUDGET is 0")
        if not self._semaphore.acquire(blocking=False):
            raise LiveError("LIVE_BUSY", "another live diagnosis is running", 429)
        counted = False
        http_client = None
        try:
            if self._store is not None:
                try:
                    self._store.reserve()
                except BudgetError:
                    raise LiveError("LIVE_BUDGET_EXHAUSTED",
                                    "persistent GPU budget is exhausted", 429) from None
                with self._lock:
                    self._counters.attempted += 1
                counted = True
            else:
                with self._lock:
                    if self._counters.attempted >= self._budget:
                        raise LiveError("LIVE_BUDGET_EXHAUSTED",
                                        f"GPU budget {self._budget} is exhausted", 429)
                    self._counters.attempted += 1
                    counted = True
            backend = (self._backend_factory(model) if self._backend_factory is not None
                       else self._remote_backend(model))
            fake_backend = isinstance(backend, FakeBackend)
            emit("connecting", "正在准备离线 Mock…" if fake_backend else "正在连接 Qwen 服务…")
            if isinstance(backend, RemoteInferenceBackend):
                http_client = getattr(backend, "_client", None)
            provider = LocalLoRAProvider(backend, template=self._template)
            provider.prepare()
            captured: dict = {}
            requested = model.upper()
            presentation: dict = {}

            async def drafter(request, history):
                results = [result for exchange in history for result in exchange.results]
                emit("calling_model", "正在生成 Mock 草稿…" if fake_backend else "正在调用 Qwen3.5-4B…")
                generation = provider.generate_draft(request.symptom, results)
                captured["generation"] = generation
                return generation.raw_output

            def adapter(diagnosis_id, content, **kwargs):
                emit("validating", "正在校验诊断结果与安全性…")
                report = adapt_sft_report(
                    diagnosis_id, content, v2_config=V2Config(enabled=v2),
                    presentation=presentation, **kwargs)
                emit("composing", "正在整理回答…")
                return report

            device = self._device_factory()
            registry = _ObservedRegistry(device, packaged_structured_index(), emit)
            workflow = DiagnosisWorkflow(
                EvidenceGroundedDraftProvider(
                    RuleBasedProvider(), drafter, label=f"Qwen3.5-4B {requested}"),
                registry, draft_adapter=adapter)
            started = time.perf_counter()
            report = asyncio.run(workflow.diagnose(DiagnosisRequest(symptom=symptom)))
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            generation = captured.get("generation")
            if generation is None:
                raise LiveError("MODEL_GENERATION_FAILED", "no generation was produced")
            raw = generation.raw_output
            guard = guard_output(raw, attempt_repair=False)
            if not guard.valid:
                with self._lock:
                    self._counters.invalid_outputs += 1
            live = bool(generation.live_inference)
            mode = (f"LIVE_QWEN_{requested}" if live else f"MOCK_QWEN_{requested}")
            device_source = "LIVE_SIMULATOR" if device.source == "live_read" else "SYNTHETIC_FIXTURE"
            meta = {
                "mode": mode,
                "live_inference": live,
                "mock": not live,
                "model": model,
                "requested_model": requested,
                "generated_at_utc": generation.generated_at_utc,
                "latency_ms": generation.latency_ms,
                "prompt_chars": generation.prompt_chars,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "request_id": generation.request_id or request_id,
                "backend": generation.backend,
                "guard": guard.as_public_dict(),
                "v2_enabled": v2,
                "v2_decision": _decision_from_limitations(report.limitations) if v2 else None,
                "v2_codes": _codes_from_limitations(report.limitations) if v2 else [],
                "safety": public_safety(raw),
                "rag_hits": _rag_hits(report),
                "device_evidence": _live_refs(report),
                "server_latency_ms": latency_ms,
                "budget": self.status(),
                "device_source": device_source,
                "sources": [device_source, mode],
            }
            answer = live_answer(report.model_dump(mode="json"), meta,
                                 question=symptom, presentation=presentation)
            meta["fault_category"] = answer["diagnosis_view"]["category"]
            with self._lock:
                self._counters.succeeded += 1
                self._last_success_utc = datetime.now(UTC).isoformat()
            self._write_audit({
                "request_id": meta.get("request_id") or uuid4().hex,
                "outcome": "success", "mode": mode, "model": model,
                "live_inference": live, "report_status": report.status,
                "v2_enabled": v2, "v2_decision": meta.get("v2_decision"),
                "guard_valid": guard.valid, "guard_error": guard.error_code,
                "generation_latency_ms": generation.latency_ms,
                "server_latency_ms": latency_ms,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "budget": self.status(), "raw_output": raw,
            })
            return {"meta": meta, "report": report.model_dump(mode="json"), "answer": answer}
        except LiveError as exc:
            with self._lock:
                self._counters.failed += 1
                self._last_error = exc.code
            self._write_audit({"request_id": request_id or uuid4().hex,
                               "outcome": "error", "code": exc.code,
                               "message": exc.message, "budget": self.status()})
            raise
        except RemoteBackendError as exc:
            with self._lock:
                self._counters.failed += 1
                self._last_error = exc.code
            self._write_audit({"request_id": request_id or uuid4().hex,
                               "outcome": "error", "code": exc.code,
                               "message": str(exc), "budget": self.status()})
            raise LiveError(exc.code, str(exc), 503) from exc
        except BackendUnavailable as exc:
            with self._lock:
                self._counters.failed += 1
                self._last_error = "MODEL_SERVICE_UNAVAILABLE"
            detail = "; ".join(getattr(exc, "blockers", ())) or str(exc)
            self._write_audit({"request_id": request_id or uuid4().hex,
                               "outcome": "error", "code": "MODEL_SERVICE_UNAVAILABLE",
                               "message": detail, "budget": self.status()})
            raise LiveError("MODEL_SERVICE_UNAVAILABLE", f"模型服务不可用：{detail}", 503) from exc
        except Exception as exc:  # noqa: BLE001 - convert to a controlled UI error
            with self._lock:
                self._counters.failed += 1
                self._last_error = type(exc).__name__
            self._write_audit({"request_id": request_id or uuid4().hex,
                               "outcome": "error", "code": "LIVE_FAILED",
                               "message": type(exc).__name__, "budget": self.status()})
            raise LiveError("LIVE_FAILED", type(exc).__name__, 503) from exc
        finally:
            if counted:
                pass
            http_client.close() if http_client is not None else None
            self._semaphore.release()

    def _write_audit(self, record: dict) -> None:
        if self._audit is not None:
            self._audit.write(record)

    def health(self) -> dict:
        """Cloud health/model-info only; never calls /v1/generate."""
        if not self.configured:
            return {"cloud_online": False, "model_ready": False,
                    "reason": self.status()["reason"], "gpu_calls": 0}
        headers = {"X-PrintDiag-Token": self._token, "Accept": "application/json"}
        try:
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=4.0) as client:
                health = client.get(self._base_url.rstrip("/") + "/healthz", headers=headers)
                info = client.get(self._base_url.rstrip("/") + "/v1/model-info", headers=headers)
            health_json = health.json() if health.status_code == 200 else {}
            info_json = info.json() if info.status_code == 200 else {}
            return {"cloud_online": health.status_code == 200,
                    "model_ready": bool(health.status_code == 200
                                        and info.status_code == 200
                                        and health_json.get("model_ready")
                                        and (info_json.get("available_modes") or [])),
                    "service": health_json.get("service"),
                    "modes": info_json.get("available_modes") or [],
                    "adapter_ready": health_json.get("adapter_ready"),
                    "http_status": health.status_code,
                    "model_info_status": info.status_code, "gpu_calls": 0}
        except Exception as exc:  # noqa: BLE001
            return {"cloud_online": False, "model_ready": False,
                    "reason": type(exc).__name__, "gpu_calls": 0}

    def _remote_backend(self, model: str) -> RemoteInferenceBackend:
        expected = None
        if model == "lora":
            try:
                expected = inspect_artifacts().adapter_sha256
            except Exception:  # noqa: BLE001 - adapter sha is optional for the demo
                expected = None

        def count_request(_request):
            with self._lock:
                self._counters.http_requests += 1

        def count_response(response):
            with self._lock:
                if response.status_code < 400:
                    self._counters.http_successes += 1
                else:
                    self._counters.http_failures += 1

        client = httpx.Client(trust_env=False, follow_redirects=False, timeout=180.0,
                              event_hooks={"request": [count_request],
                                           "response": [count_response]})
        return RemoteInferenceBackend(self._base_url, self._token, model=model,
                                      timeout_s=180.0, expected_adapter_sha256=expected,
                                      client=client)


# BackendUnavailable is imported above for controlled cloud failures.


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _decision_from_limitations(limitations) -> str | None:
    codes = _codes_from_limitations(limitations)
    if any(code in ("ENDSTOP_MISMATCH_OBSERVED", "THERMAL_FAULT_OBSERVED") for code in codes):
        return "supported"
    if any(code.startswith("NORMAL_") or code == "NO_MATCHING_ANOMALY" for code in codes):
        return "contradicted"
    if any(code in ("NO_TOOL_RECEIPT", "DEVICE_BOUNDARY",
                    "IDLE_DOES_NOT_EVIDENCE_MOTION_FAULT") for code in codes):
        return "hypothesis_only"
    return "insufficient" if codes else None


def _codes_from_limitations(limitations) -> list[str]:
    codes = []
    for item in limitations:
        detail = getattr(item, "detail", item.get("detail", "") if isinstance(item, dict) else "")
        marker = "V2 编码："
        if marker in detail:
            codes.append(detail.split(marker, 1)[1].strip())
    return codes


def _rag_hits(report) -> list[dict]:
    hits = []
    for rank, ref in enumerate(report.evidence_refs, start=1):
        if ref.source != "curated_knowledge":
            continue
        data = ref.data
        hit = {
            "document_id": data.document_id,
            "chunk_id": getattr(data, "chunk_id", None),
            "excerpt": data.excerpt,
            "source_reference": getattr(data, "source_reference", None),
            "source_title": getattr(data, "title", None),
            "source_type": "curated_knowledge",
            "retrieval_score": getattr(data, "retrieval_score", None),
            "retrieval_rank": rank,
            "knowledge_version": getattr(data, "revision", None),
        }
        hits.append(hit)
    return hits


def _live_refs(report) -> list[dict]:
    refs = []
    for ref in report.evidence_refs:
        if ref.source != "live_read":
            continue
        refs.append({"tool": ref.tool, "evidence_id": ref.evidence_id,
                     "observed_at_utc": ref.observed_at_utc.isoformat()
                     if ref.observed_at_utc else None,
                     "data": ref.data.model_dump(mode="json")
                     if hasattr(ref.data, "model_dump") else ref.data})
    return refs
