"""Public snapshot smoke without a simulator, model weights, or training data."""

import json

from fastapi.testclient import TestClient

from scripts.run_console_demo import DEMO_SYMPTOM, build_demo


def test_console_demo_round_trip(monkeypatch):
    import httpx

    def no_network(*args, **kwargs):
        raise AssertionError("the offline demo must not use a real HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    # The launcher owns these process-local values. Restore them after the test.
    for key in ("PRINTDIAG_HISTORY_DB", "PRINTDIAG_ALLOW_REMOTE_INFERENCE",
                "PRINTDIAG_LIVE_GPU_BUDGET", "PRINTDIAG_SIM_CONTROL_ENABLED",
                "PRINTDIAG_LIVE_BUDGET_STATE", "PRINTDIAG_LIVE_AUDIT_DIR"):
        monkeypatch.delenv(key, raising=False)
    app, _ = build_demo()
    with TestClient(app) as client:
        status = client.get("/api/device/status").json()
        assert status["source"] == "SYNTHETIC_FIXTURE"
        assert status["device_connected"] is False
        live = client.get("/api/live/status").json()
        assert live["mock"] and not live["live_inference"] and not live["cloud_configured"]
        result = client.post("/api/ask/stream", json={"symptom": DEMO_SYMPTOM, "v2": True})
        final = json.loads(result.text.splitlines()[-1])
        view = final["answer"]["diagnosis_view"]
        assert view["title"] == "疑似挤出不足"
        assert view["status_code"] == "insufficient_evidence"
        assert view["technical"]["references"] and not view["technical"]["receipts"]
        blocked = client.post("/api/demo/diagnose", json={
            "scenario_id": "safety", "symptom": DEMO_SYMPTOM, "v2": True,
        })
        assert blocked.json()["meta"]["safety"]["status"] == "blocked"
        assert "Disable thermal protection" not in blocked.text
        key = final["history"]["diagnosis_id"]
        exported = client.get(f"/api/history/{key}/report")
        assert "疑似挤出不足" in exported.text
        assert "技术详情" in exported.text
