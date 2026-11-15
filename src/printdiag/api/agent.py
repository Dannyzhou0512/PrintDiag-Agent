"""FastAPI entry point. Mock device is the default; the local simulator is opt-in."""

import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from printdiag.agents.live_provider import LiveProviderSettings
from printdiag.agents.providers import MockProvider
from printdiag.agents.symptom_routing import RuleBasedProvider
from printdiag.agents.workflow import DiagnosisWorkflow
from printdiag.rag.maintenance import packaged_index
from printdiag.rag.structured import packaged_structured_index
from printdiag.schemas.agent import DiagnosisReport, DiagnosisRequest
from printdiag.tools.device_reads import DeviceReads, simulator_device, simulator_origin
from printdiag.tools.registry import ToolRegistry


def create_device() -> DeviceReads:
    """Select the device source. Mock stays the default; the simulator is opt-in."""
    mode = os.environ.get("PRINTDIAG_DEVICE_SOURCE", "mock")
    if mode == "mock":
        return DeviceReads()
    if mode == "simulator":
        simulator_origin(os.environ)  # raises unless explicitly authorized and pinned
        return simulator_device()
    raise ValueError(
        "live device reads are disabled: only mock, or simulator with "
        "PRINTDIAG_ALLOW_SIMULATOR=1 and http://127.0.0.1:8100, are allowed"
    )


def create_app(*, workflow: DiagnosisWorkflow | None = None) -> FastAPI:
    if workflow is None:
        device = create_device()
        mode = os.environ.get("PRINTDIAG_LLM_PROVIDER", "mock")
        if mode == "mock":
            provider = RuleBasedProvider() if device.source == "live_read" else MockProvider()
        elif mode == "rule":
            provider = RuleBasedProvider()
        elif mode == "openai_compatible":
            provider = LiveProviderSettings.from_environment().for_diagnosis
        else:
            raise ValueError("unknown LLM Provider mode")
        # Structured BM25 corpus for the simulator/agent path; the legacy v0.1
        # index stays the default for the pure mock path (backward compatible).
        index = packaged_structured_index() if device.source == "live_read" else packaged_index()
        workflow = DiagnosisWorkflow(provider, ToolRegistry(device, index))

    app = FastAPI(title="PrintDiag Agent", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": {"code": "INVALID_ARGUMENT"}})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "knowledge": "available" if workflow.knowledge_available else "unavailable"}

    @app.post("/v1/diagnoses", response_model=DiagnosisReport)
    async def diagnose(request: DiagnosisRequest) -> DiagnosisReport:
        return await workflow.diagnose(request)

    return app
