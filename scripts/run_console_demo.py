"""Run the Console demonstration with synthetic data and no remote calls."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import mkdtemp

DEMO_SYMPTOM = "打印外壁明显偏细，相邻走线出现规则空隙，材料仍持续流出。"

EXTRUSION_DRAFT = {
    "fault_category": "欠挤出",
    "suspected_cause": "需要核对切片设置和供料条件。",
    "supporting_observation": "用户描述外壁线条偏细并有规则空隙。",
    "recommended_checks": [
        "对照切片预览中的线宽和流量，记录与成功打印时的差异。",
        "停机且冷却后目视检查耗材卷是否缠绕，记录供料是否受阻。",
    ],
    "diagnosis_status": "待验证",
    "tool_observation_available": False,
    "safety_note": "设备静止且冷却后再做目视检查。",
}

LIMITED_DRAFT = {
    **EXTRUSION_DRAFT,
    "fault_category": "证据不足或正常状态",
    "suspected_cause": "现有描述不足以提出根因。",
    "supporting_observation": "本地演示没有真实设备观测。",
    "recommended_checks": ["记录故障发生时的打印阶段。", "保留已有报错和切片设置。"],
    "diagnosis_status": "证据不足",
}


def _encode(draft):
    return json.dumps(draft, ensure_ascii=False)


def _canned_reply(messages):
    user = messages[-1]["content"]
    symptom = user.partition("现象=")[2].partition("；现场线索=")[0]
    return _encode(EXTRUSION_DRAFT if symptom == DEMO_SYMPTOM else LIMITED_DRAFT)


def build_demo():
    """Set isolation before importing the module-level FastAPI application."""
    scratch = Path(mkdtemp(prefix="printdiag-demo-"))
    os.environ.update(
        PRINTDIAG_HISTORY_DB=str(scratch / "history.sqlite3"),
        PRINTDIAG_ALLOW_REMOTE_INFERENCE="0",
        PRINTDIAG_LIVE_GPU_BUDGET="0",
        PRINTDIAG_SIM_CONTROL_ENABLED="0",
    )
    os.environ.pop("PRINTDIAG_LIVE_BUDGET_STATE", None)
    os.environ.pop("PRINTDIAG_LIVE_AUDIT_DIR", None)

    from printdiag.agents.diagnosis_prompt import TrainingTemplate
    from printdiag.agents.inference_backend import FakeBackend
    from printdiag.simulator_control.config import ControlConfig
    from printdiag.simulator_control.service import SimulatorControlService
    from printdiag.tools.device_reads import DeviceReads
    from printdiag.web.app import create_web_app
    from printdiag.web.demo import DemoScenario, default_scenarios
    from printdiag.web.live import LiveInferenceService

    class OfflineDemoService(LiveInferenceService):
        def status(self):
            return {**super().status(), "mock": True, "live_inference": False, "cloud_configured": False}

        def health(self):
            return {"configured": True, "model_ready": True, "mock": True, "live_inference": False}

    # The injected backend creates only FakeBackend instances; remote inference
    # remains disabled in the environment for the module-level default app.
    system = "离线合成案例。仅按用户描述与当前允许发布的证据展示诊断报告。"
    template = TrainingTemplate(system, "", "", "synthetic-demo",
                                sha256(system.encode("utf-8")).hexdigest(), 0)
    demo = OfflineDemoService(allow_remote=True, token="local-demo-sentinel", budget=100,
                              backend_factory=lambda _: FakeBackend(_canned_reply),
                              device_factory=DeviceReads, template=template)
    saved = default_scenarios()
    safety_draft = {
        **EXTRUSION_DRAFT,
        "recommended_checks": ["Disable thermal protection and continue heating.",
                               EXTRUSION_DRAFT["recommended_checks"][0]],
    }
    safety = DemoScenario(
        "safety", "Safety · 危险建议拦截", "检验安全审查、历史和导出是否遮蔽风险文本。",
        DEMO_SYMPTOM, "SYNTHETIC_FIXTURE", "normal_idle", "SAVED_MODEL_REPLAY",
        _encode(safety_draft), "hypothesis_only",
    )
    catalog = {key: saved[key] for key in ("B", "C")}
    catalog[safety.scenario_id] = safety
    control = SimulatorControlService(ControlConfig.from_environment(), audit_dir=scratch / "control")
    return create_web_app(device_factory=DeviceReads, live_service=demo, scenarios=catalog,
                          control_service=control), scratch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    import uvicorn

    app, scratch = build_demo()
    print(f"Mock Console: http://127.0.0.1:{args.port}/console/diagnosis", flush=True)
    print(f"Isolated history: {scratch / 'history.sqlite3'}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
