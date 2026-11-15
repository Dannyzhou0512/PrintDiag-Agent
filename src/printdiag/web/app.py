"""FastAPI app for the PrintDiag assistant (P10-7B).

Live Qwen, status-only questions and saved-model demos are separate paths.
The server never calls the model at startup; GPU budget defaults to 0.
"""

from pathlib import Path
import json
import queue
import threading
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response
from pydantic import BaseModel, Field

from printdiag.rag.structured import load_builtin_chunks, packaged_structured_index
from printdiag.history import HistoryError, HistoryService
from printdiag.reporting.diagnosis_view import history_view
from printdiag.simulator_control.config import ControlConfig
from printdiag.simulator_control.schemas import ArmRequest, JogRequest
from printdiag.simulator_control.service import SimulatorControlService
from printdiag.web.answers import device_answer, intent_answer
from printdiag.web.demo import (
    DemoScenario, default_scenarios, device_status, markdown_report, run_scenario,
)
from printdiag.web.intent import Intent, route_intent
from printdiag.web.live import LiveError, LiveInferenceService

WEB_DIR = Path(__file__).resolve().parent
INDEX = WEB_DIR / "index.html"
CONSOLE_DIR = WEB_DIR / "console"
CONSOLE_PAGES = {"overview", "diagnosis", "simulator", "events", "history", "knowledge", "settings"}


class DemoRequest(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=20)
    symptom: str = Field(min_length=1, max_length=500)
    v2: bool = False


class AskRequest(BaseModel):
    symptom: str = Field(min_length=1, max_length=500)
    model: str = Field(default="base", pattern="^(base|lora)$")
    v2: bool = False
    mode: str = Field(default="auto", pattern="^(auto|live)$")


class LiveRequest(BaseModel):
    symptom: str = Field(min_length=1, max_length=500)
    model: str = Field(default="base", pattern="^(base|lora)$")
    v2: bool = False


def create_web_app(*, device_factory=None, scenarios: dict[str, DemoScenario] | None = None,
                   live_service: LiveInferenceService | None = None,
                   control_service: SimulatorControlService | None = None,
                   history_service: HistoryService | None = None) -> FastAPI:
    catalog = scenarios or default_scenarios()
    live = live_service or LiveInferenceService()
    control = control_service or SimulatorControlService(ControlConfig.from_environment())
    history = history_service or HistoryService.from_environment()
    app = FastAPI(title="PrintDiag Assistant", version="p10-8c")
    factory = device_factory

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "mode": "assistant", "version": "p10-8c",
                "live": live.status(), "gpu_requests_this_process": live.status()["used"],
                "history": {"schema_version": 1, "enabled": True}}

    @app.get("/api/scenarios")
    def scenarios_api() -> dict:
        return {"scenarios": [
            {"scenario_id": item.scenario_id, "title": item.title,
             "description": item.description, "symptom": item.symptom,
             "device_source": item.device_source, "model_source": item.model_source,
             "expected_v2": item.expected_v2}
            for item in catalog.values()]}

    @app.get("/api/device/status")
    def device_api() -> dict:
        return device_status(factory)

    @app.get("/api/live/status")
    def live_api() -> dict:
        return live.status()

    @app.get("/api/live/health")
    def live_health_api() -> dict:
        return live.health()

    @app.post("/api/live/diagnose")
    def live_diagnose_api(request: LiveRequest) -> dict:
        try:
            result = live.diagnose(symptom=request.symptom.strip(), model=request.model,
                                   v2=request.v2)
            result["history"] = persist_diagnosis(
                result, question=request.symptom, intent="fault_diagnosis")
            return result
        except LiveError as exc:
            persist_failure(question=request.symptom, intent="fault_diagnosis",
                            source_mode="LIVE_MODEL", error_code=exc.code)
            raise HTTPException(status_code=exc.status_code,
                                detail={"code": exc.code, "message": exc.message}) from exc

    @app.post("/api/ask")
    def ask_api(request: AskRequest) -> dict:
        symptom = request.symptom.strip()
        route = route_intent(symptom)
        if route.intent in (Intent.GREETING, Intent.MODEL_IDENTITY,
                            Intent.CAPABILITIES, Intent.CLARIFICATION_NEEDED):
            answer = intent_answer(route, live.status())
            return {"mode": answer["mode"], "answer": answer,
                    "route": {"intent": route.intent.value, "reason": route.reason}}
        if route.intent is Intent.DEVICE_STATUS:
            status = device_status(factory)
            return {"mode": "DEVICE_STATUS_ONLY", "answer": device_answer(status),
                    "route": {"intent": route.intent.value, "reason": route.reason}}
        try:
            result = live.diagnose(symptom=symptom, model=request.model, v2=request.v2)
            result["history"] = persist_diagnosis(
                result, question=symptom, intent=route.intent.value)
            return result
        except LiveError as exc:
            persist_failure(question=symptom, intent=route.intent.value,
                            source_mode="LIVE_MODEL", error_code=exc.code)
            raise HTTPException(status_code=exc.status_code,
                                detail={"code": exc.code, "message": exc.message}) from exc

    @app.post("/api/ask/stream")
    def ask_stream_api(request: AskRequest) -> StreamingResponse:
        symptom = request.symptom.strip()
        route = route_intent(symptom)
        started = time.perf_counter()

        def event(payload: dict) -> bytes:
            return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

        def stage(key: str, label: str) -> bytes:
            return event({"type": "stage", "stage": key, "label": label,
                          "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})

        def generate():
            yield stage("understanding", "正在理解问题…")
            if route.intent in (Intent.GREETING, Intent.MODEL_IDENTITY,
                                Intent.CAPABILITIES, Intent.CLARIFICATION_NEEDED):
                yield stage("composing", "正在整理回答…")
                answer = intent_answer(route, live.status())
                yield event({"type": "final", "mode": answer["mode"], "answer": answer,
                             "route": {"intent": route.intent.value, "reason": route.reason},
                             "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})
                return
            if route.intent is Intent.DEVICE_STATUS:
                yield stage("reading_device", "正在读取设备状态…")
                status = device_status(factory)
                yield stage("composing", "正在整理回答…")
                answer = device_answer(status)
                yield event({"type": "final", "mode": "DEVICE_STATUS_ONLY", "answer": answer,
                             "route": {"intent": route.intent.value, "reason": route.reason},
                             "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})
                return
            messages: queue.Queue = queue.Queue()

            def callback(key, label):
                messages.put(("stage", key, label))

            def worker():
                try:
                    result = live.diagnose(symptom=symptom, model=request.model,
                                           v2=request.v2, stage_callback=callback)
                    result["history"] = persist_diagnosis(
                        result, question=symptom, intent=route.intent.value)
                    messages.put(("final", result))
                except Exception as exc:  # noqa: BLE001 - converted to an error event
                    persist_failure(question=symptom, intent=route.intent.value,
                                    source_mode="LIVE_MODEL",
                                    error_code=getattr(exc, "code", "LIVE_FAILED"))
                    messages.put(("error", exc))

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            while True:
                try:
                    item = messages.get(timeout=1.0)
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    continue
                if item[0] == "stage":
                    yield stage(item[1], item[2])
                elif item[0] == "final":
                    payload = item[1]
                    payload["route"] = {"intent": route.intent.value, "reason": route.reason}
                    payload["streaming"] = "stage-events-only"
                    yield event({"type": "final", **payload,
                                 "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})
                    return
                else:
                    exc = item[1]
                    yield event({"type": "error", "code": getattr(exc, "code", "LIVE_FAILED"),
                                 "message": str(exc),
                                 "status": getattr(exc, "status_code", 503),
                                 "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})
                    return

        return StreamingResponse(generate(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    def require_local_operator(request: Request) -> None:
        host = request.client.host if request.client else ""
        if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
            raise HTTPException(status_code=403, detail={"code": "LOCAL_OPERATOR_ONLY"})
        origin = request.headers.get("origin")
        if origin:
            try:
                origin_host = urlsplit(origin).hostname or ""
            except ValueError:
                origin_host = ""
            if origin_host not in {"127.0.0.1", "::1", "localhost"}:
                raise HTTPException(status_code=403, detail={"code": "LOCAL_ORIGIN_ONLY"})

    def persist_diagnosis(result: dict, *, question: str, intent: str | None,
                          source_mode: str | None = None) -> dict:
        """Persist after diagnosis; a storage failure never changes the result."""
        try:
            saved = history.save_result(result, question=question, intent=intent,
                                        source_mode=source_mode)
            return {"saved": True, "created": saved.created,
                    "diagnosis_id": saved.diagnosis_id}
        except HistoryError as exc:
            return {"saved": False, "code": exc.code}
        except Exception:  # noqa: BLE001 - persistence must not mask diagnosis
            return {"saved": False, "code": "HISTORY_WRITE_FAILED"}

    def persist_failure(*, question: str, intent: str | None, source_mode: str,
                        error_code: str) -> None:
        try:
            history.save_failure(question=question, intent=intent,
                                 source_mode=source_mode, error_code=error_code)
        except Exception:  # noqa: BLE001 - failure metadata is best effort
            pass

    @app.get("/api/history")
    def history_list_api(request: Request, limit: int = Query(default=20, ge=1, le=100),
                         offset: int = Query(default=0, ge=0),
                         status: str | None = Query(default=None,
                                                    pattern="^(complete|partial|insufficient_evidence|failed)$"),
                         source_mode: str | None = Query(default=None,
                                                         pattern="^(LIVE_MODEL|SAVED_REPLAY|MOCK_TEST|DEVICE_ONLY)$"),
                         model: str | None = Query(default=None, pattern="^(base|lora)$"),
                         q: str | None = Query(default=None, max_length=200)) -> dict:
        require_local_operator(request)
        page = history.list(limit=limit, offset=offset, status=status,
                            source_mode=source_mode, model=model, text=q)
        # List responses intentionally omit the large validated report/answer.
        rows = []
        for item in page["items"]:
            view = history_view(item)
            row = {key: value for key, value in item.items()
                   if key not in {"validated_report", "answer"}}
            symptom = next((entry["text"] for entry in view["basis"]
                            if entry["source"] == "用户描述"), "")
            excerpt = f"{symptom[:95]}…" if len(symptom) > 96 else symptom
            row["display"] = {"title": view["title"], "excerpt": excerpt,
                              "safety": {"label": view["safety"]["label"],
                                         "status": view["safety"]["status"]}}
            rows.append(row)
        page["items"] = rows
        return page

    @app.get("/api/history/{diagnosis_id}/report", response_class=PlainTextResponse)
    def history_report_api(diagnosis_id: str, request: Request) -> PlainTextResponse:
        require_local_operator(request)
        text = history.report_markdown(diagnosis_id)
        if text is None:
            raise HTTPException(status_code=404, detail={"code": "HISTORY_NOT_FOUND"})
        return PlainTextResponse(text, media_type="text/markdown; charset=utf-8",
                                 headers={"Content-Disposition":
                                          f'attachment; filename="{diagnosis_id}.md"'})

    @app.get("/api/history/{diagnosis_id}")
    def history_detail_api(diagnosis_id: str, request: Request) -> dict:
        require_local_operator(request)
        record = history.get(diagnosis_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "HISTORY_NOT_FOUND"})
        return {"record": {**record, "diagnosis_view": history_view(record)}}

    @app.get("/api/simulator/control/status")
    def simulator_control_status(request: Request) -> dict:
        require_local_operator(request)
        return control.status()

    @app.post("/api/simulator/control/arm")
    def simulator_control_arm(request: ArmRequest, raw_request: Request) -> dict:
        require_local_operator(raw_request)
        return control.arm(confirm_local_simulator=request.confirm_local_simulator)

    @app.post("/api/simulator/control/jog")
    def simulator_control_jog(request: JogRequest, raw_request: Request) -> dict:
        require_local_operator(raw_request)
        return control.jog(axis=request.axis, direction=request.direction, step=request.step)

    @app.post("/api/simulator/control/disarm")
    def simulator_control_disarm(raw_request: Request) -> dict:
        require_local_operator(raw_request)
        return control.disarm()

    @app.post("/api/simulator/control/hold/renew")
    def simulator_control_hold_renew(raw_request: Request) -> dict:
        require_local_operator(raw_request)
        return control.renew_hold()

    @app.get("/api/simulator/control/history")
    def simulator_control_history(request: Request, limit: int = Query(default=30, ge=1, le=100)) -> dict:
        require_local_operator(request)
        return {"records": control.history(limit)}

    @app.get("/api/knowledge/overview")
    def knowledge_overview_api() -> dict:
        """Read-only BM25 inventory; no file writes or management actions."""
        index = packaged_structured_index()
        documents = []
        for chunk in load_builtin_chunks():
            documents.append({
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "fault_category": chunk.fault_category,
                "applicable_device": chunk.applicable_device,
                "source_reference": chunk.source_reference,
                "excerpt": chunk.excerpt,
                "recommended_checks": list(chunk.recommended_checks),
                "limitations": list(chunk.limitations),
            })
        return {"available": index.available, "size": index.size,
                "categories": list(index.categories()), "documents": documents}

    @app.get("/api/knowledge/search")
    def knowledge_search_api(q: str = Query(min_length=1, max_length=120),
                             limit: int = Query(default=3, ge=1, le=10)) -> dict:
        index = packaged_structured_index()
        hits = index.search(q, limit)
        return {"query": q, "hits": [hit.model_dump(mode="json") for hit in hits]}

    app.mount("/console/static", NoCacheStaticFiles(directory=CONSOLE_DIR), name="console-static")

    @app.get("/console", response_class=HTMLResponse)
    def console_root() -> RedirectResponse:
        return RedirectResponse(url="/console/overview", status_code=307)

    @app.get("/console/{page}", response_class=HTMLResponse)
    def console_page(page: str) -> HTMLResponse:
        if page not in CONSOLE_PAGES:
            raise HTTPException(status_code=404, detail={"code": "UNKNOWN_CONSOLE_PAGE"})
        html = (CONSOLE_DIR / "console.html").read_text(encoding="utf-8")
        # First paint uses the same read-only snapshot as /api/device/status.
        # This never calls Qwen and degrades to offline on read failure.
        try:
            boot_device = device_status(factory)
        except Exception:
            boot_device = {"online": False, "error": "device status unavailable",
                           "observed_at_utc": None}
        boot_json = json.dumps(boot_device, ensure_ascii=False).replace("<", "\\u003c")
        html = html.replace("</head>",
                            f'<script>window.__PDC_BOOT_DEVICE__ = {boot_json};</script></head>')
        return HTMLResponse(html)

    @app.post("/api/demo/diagnose")
    def diagnose_api(request: DemoRequest) -> dict:
        scenario = catalog.get(request.scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail={"code": "UNKNOWN_SCENARIO"})
        symptom = request.symptom.strip()
        if symptom != scenario.symptom.strip():
            raise HTTPException(status_code=422, detail={
                "code": "REPLAY_SYMPTOM_MISMATCH",
                "message": "Replay 只能演示固定场景；自由提问请使用 Live/状态问答。"})
        try:
            payload = run_scenario(scenario, symptom=symptom, v2=request.v2,
                                   device_factory=factory)
            payload["history"] = persist_diagnosis(
                payload, question=symptom, intent="saved_replay",
                source_mode="SAVED_REPLAY")
            return payload
        except Exception as exc:  # noqa: BLE001 - return a controlled UI error
            raise HTTPException(status_code=503, detail={
                "code": "DEMO_UNAVAILABLE", "message": type(exc).__name__}) from exc

    @app.post("/api/demo/markdown", response_class=PlainTextResponse)
    def markdown_api(request: DemoRequest) -> PlainTextResponse:
        scenario = catalog.get(request.scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail={"code": "UNKNOWN_SCENARIO"})
        if request.symptom.strip() != scenario.symptom.strip():
            raise HTTPException(status_code=422, detail={"code": "REPLAY_SYMPTOM_MISMATCH"})
        payload = run_scenario(scenario, symptom=request.symptom.strip(),
                               v2=request.v2, device_factory=factory)
        return PlainTextResponse(markdown_report(payload),
                                 media_type="text/markdown; charset=utf-8")

    return app


app = create_web_app()
