"""Separate HTTP client for simulator write control.

This client is intentionally NOT a MarlinClient and NOT registered in any Agent
tool registry. It exposes only the narrow methods needed by the human-operated
jog service; arbitrary G-code cannot be submitted through the web API.
"""

import re

import httpx

from printdiag.simulator_control.config import SIMULATOR_URL

_ALLOWED_G1 = re.compile(r"^G1 [XYZ]-?\d+(?:\.\d{1,3})? F\d+(?:\.\d+)?$")


class ControlClientError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class SimulatorControlClient:
    def __init__(self, base_url: str = SIMULATOR_URL, *, timeout_s: float = 5.0,
                 client: httpx.Client | None = None) -> None:
        url = httpx.URL(base_url)
        if url.host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("simulator control requires a loopback URL")
        self.base_url = str(url).rstrip("/")
        self._client = client or httpx.Client(trust_env=False, follow_redirects=False,
                                              timeout=timeout_s)
        self._owns = client is None

    def close(self) -> None:
        if self._owns:
            self._client.close()

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self._client.request(method, self.base_url + path, **kwargs)
        except httpx.TimeoutException:
            raise ControlClientError("CONTROL_TIMEOUT", "simulator request timed out") from None
        except httpx.HTTPError as exc:
            raise ControlClientError("SIMULATOR_UNAVAILABLE", type(exc).__name__) from exc
        if response.status_code != 200:
            raise ControlClientError("SIMULATOR_HTTP_ERROR",
                                     f"simulator returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            raise ControlClientError("INVALID_SIMULATOR_RESPONSE", "response is not JSON") from None
        if not isinstance(payload, dict):
            raise ControlClientError("INVALID_SIMULATOR_RESPONSE", "response is not an object")
        return payload

    def ping(self) -> dict:
        return self._request("GET", "/ping")

    def components(self) -> dict:
        return self._request("GET", "/components")

    def state(self) -> dict:
        return self._request("GET", "/state")

    def idle(self) -> dict:
        return self._request("GET", "/idle")

    def post_g1(self, command: str) -> dict:
        if not _ALLOWED_G1.fullmatch(command):
            raise ControlClientError("GCODE_NOT_ALLOWED", "only a single generated G1 jog is allowed")
        return self._request("POST", "/gcode", json={"command": command})

    # P10-8B-1 firmware-aware protocol. These endpoints are simulator-only and
    # loopback-only; the Python service never sends arbitrary G-code.
    def firmware_state(self) -> dict:
        return self._request("GET", "/agent/firmware-motion-state")

    def session_acquire(self) -> dict:
        return self._request("POST", "/agent/control/session", json={})

    def session_release(self, session_id: str) -> dict:
        return self._request("POST", "/agent/control/release", json={"session_id": session_id})

    def control_status(self) -> dict:
        return self._request("GET", "/agent/control/status")

    def control_jog(self, session_id: str, axis: str, positive: bool, step: float) -> dict:
        return self._request("POST", "/agent/control/jog", json={
            "session_id": session_id, "axis": axis,
            "positive": bool(positive), "step": float(step),
        })

    def hold_acquire(self, lease_ms: int = 120000) -> dict:
        return self._request("POST", "/agent/control/hold", json={
            "confirm_local_simulator": True, "lease_ms": int(lease_ms),
        })

    def hold_renew(self, hold_id: int, lease_ms: int = 120000) -> dict:
        return self._request("POST", "/agent/control/hold/renew", json={
            "hold_id": int(hold_id), "lease_ms": int(lease_ms),
        })

    def hold_release(self, hold_id: int) -> dict:
        return self._request("POST", "/agent/control/hold/release", json={
            "hold_id": int(hold_id),
        })
