"""Cloud inference FastAPI service (bind 127.0.0.1:8008, token authenticated).

Endpoints:
  GET  /healthz          public liveness/model readiness summary
  GET  /v1/model-info    authenticated model + adapter metadata
  GET  /v1/models        authenticated inventory (compatibility alias)
  GET  /v1/status        authenticated GPU/loaded status
  POST /v1/generate      authenticated text generation (mode=base|lora)

The service never exposes arbitrary paths, G-code, device control or downloads.
Tokens are read from the environment, compared in constant time and never logged.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from cloud_inference.engine import EngineSettings, EngineUnavailable, HfEngine
from cloud_inference.schemas import (
    GenerateRequest, GenerateResponse, HealthResponse, ModelInfoResponse,
    ModelMetadata,
)
from cloud_inference.security import load_token_from_env, token_matches

SERVICE_NAME = "printdiag-cloud-inference"
SERVICE_VERSION = "1.0.0"
MAX_BODY_BYTES = 1_000_000
GENERATION_TIMEOUT_S = 240.0

logger = logging.getLogger("printdiag.cloud")


def create_cloud_app(*, token: str | None = None, engine: HfEngine | Any | None = None,
                     service_name: str = SERVICE_NAME, eager_load: bool = True,
                     generation_timeout_s: float = GENERATION_TIMEOUT_S) -> FastAPI:
    secret = token if token is not None else load_token_from_env()
    runtime = engine if engine is not None else HfEngine(
        EngineSettings.from_environment(dict(__import__("os").environ)))
    semaphore = asyncio.Semaphore(1)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if eager_load and hasattr(runtime, "warmup"):
            await asyncio.to_thread(runtime.warmup)
        yield
        unload = getattr(runtime, "unload", None)
        if callable(unload):
            await asyncio.to_thread(unload)

    app = FastAPI(title=service_name, version=SERVICE_VERSION, lifespan=lifespan)
    app.state.engine = runtime
    app.state.semaphore = semaphore
    app.state.secret = secret  # never logged or returned

    def require_token(x_printdiag_token: str | None = Header(default=None),
                      authorization: str | None = Header(default=None)) -> None:
        candidate = x_printdiag_token
        if candidate is None and authorization:
            candidate = authorization.partition(" ")[2].strip() or None
        if not token_matches(candidate, secret):
            raise HTTPException(status_code=401, detail={"code": "REMOTE_AUTH_FAILED"})

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > MAX_BODY_BYTES:
                    return JSONResponse(status_code=413,
                                        content={"detail": {"code": "PAYLOAD_TOO_LARGE"}})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": {"code": "BAD_REQUEST"}})
        started = time.perf_counter()
        response = await call_next(request)
        logger.info("%s %s -> %s in %.1f ms", request.method, request.url.path,
                    response.status_code, (time.perf_counter() - started) * 1000)
        return response

    def _variants() -> dict[str, dict[str, Any]]:
        return runtime.variants()

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        variants = _variants()
        base = variants.get("base", {})
        lora = variants.get("lora", {})
        model_ready = bool(base.get("loaded") or (not eager_load and base.get("configured")))
        adapter_ready = bool(lora.get("loaded") or (not eager_load and lora.get("configured")))
        return HealthResponse(
            ok=True, service_status="ok", service=service_name, service_version=SERVICE_VERSION,
            model_ready=model_ready, adapter_ready=adapter_ready)

    @app.get("/v1/model-info", response_model=ModelInfoResponse,
             dependencies=[Depends(require_token)])
    async def model_info() -> ModelInfoResponse:
        variants = _variants()
        metadata = runtime.model_metadata() if hasattr(runtime, "model_metadata") else {}
        return ModelInfoResponse(
            model_name=metadata.get("model_name", "Qwen3.5-4B"),
            architecture=metadata.get("architecture"),
            adapter_loaded=bool(variants.get("lora", {}).get("loaded")),
            adapter_sha256=variants.get("lora", {}).get("adapter_sha256"),
            available_modes=[name for name, value in variants.items() if value.get("configured")],
            inference_dtype=metadata.get("inference_dtype", "bfloat16"),
            service_version=SERVICE_VERSION, service=service_name)

    @app.get("/v1/models", dependencies=[Depends(require_token)])
    async def models_alias() -> dict[str, Any]:
        return {"service": service_name, "variants": _variants()}

    @app.get("/v1/status", dependencies=[Depends(require_token)])
    async def status() -> dict[str, Any]:
        return {"service": service_name, "gpu": runtime.gpu_status(), "variants": _variants()}

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest,
                       _: None = Depends(require_token)) -> GenerateResponse:
        request_id = request.request_id or uuid4().hex
        messages = [{"role": item.role, "content": item.content} for item in request.messages]
        options = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
            "temperature": request.temperature if request.temperature is not None else 0.0,
            "top_p": request.inference_options.top_p,
            "seed": request.inference_options.seed,
            "enable_thinking": request.inference_options.enable_thinking,
        }
        started = time.perf_counter()
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(runtime.generate, request.mode, messages, options),
                    timeout=generation_timeout_s)
            except EngineUnavailable as exc:
                raise HTTPException(status_code=503,
                                    detail={"code": "MODEL_NOT_READY",
                                            "message": str(exc)}) from None
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504,
                                    detail={"code": "GENERATION_TIMEOUT"}) from None
            except Exception as exc:  # noqa: BLE001 - never leak internal traces
                logger.warning("generation failed request_id=%s mode=%s type=%s",
                               request_id, request.mode, type(exc).__name__)
                raise HTTPException(status_code=500,
                                    detail={"code": "MODEL_GENERATION_FAILED"}) from None
        metadata = runtime.model_metadata() if hasattr(runtime, "model_metadata") else {}
        return GenerateResponse(
            request_id=request_id, mode=result["variant"], raw_output=result["text"],
            input_tokens=result.get("input_tokens"), output_tokens=result.get("output_tokens"),
            inference_latency_ms=float(result.get("latency_ms", (time.perf_counter() - started) * 1000)),
            model_metadata=ModelMetadata(
                model_name=metadata.get("model_name", "Qwen3.5-4B"),
                architecture=metadata.get("architecture"),
                adapter_sha256=result.get("adapter_sha256") or metadata.get("adapter_sha256"),
                inference_dtype=metadata.get("inference_dtype", "bfloat16"),
                mode=result["variant"]),
            live_inference=True, error=None)

    return app


# `from cloud_inference.service import app` (production); None when unconfigured.
try:
    app: FastAPI | None = create_cloud_app()
except ValueError:
    app = None
