"""Agent-only GET transport and explicit device projections."""

from collections.abc import Callable, Mapping
from math import isfinite
from threading import Event
from time import monotonic
from typing import Literal
from urllib.parse import quote

import httpx

from printdiag.clients.marlin_client import (
    MarlinClient, MarlinError, MarlinHTTPError, MarlinProtocolError, MarlinTimeoutError,
)
from printdiag.config import Config
from printdiag.schemas.agent import (
    Axis, EndstopComponentProjection, EndstopData, ErrorCode, Issue, PrinterStateProjection, XYZ,
)

ENDSTOPS: dict[Axis, str] = {"x_min": "Endstop(X Min)", "y_min": "Endstop(Y Min)", "z_min": "Endstop(Z Min)"}
# Temperature components are already inside the single GET /state payload, so the
# projection needs no extra endpoint and the read-only whitelist stays unchanged.
HEATERS: dict[str, str] = {"hotend": "Hotend0 Heater", "bed": "Bed Heater"}
GET_PATHS = {b"/state", b"/idle", *(f"/state/{quote(name, safe='')}".encode() for name in ENDSTOPS.values())}
MOCK_ORIGIN = "http://printer.mock.invalid"
AUTHORIZED_SIMULATOR_ORIGIN = "http://127.0.0.1:8100"
DeviceProvenance = Literal["mock", "simulator"]
MAX_BODY_BYTES = 1024 * 1024


class AccessDenied(MarlinError):
    """An HTTP request or redirect violated the Agent's read policy."""


def device_origin(value: str) -> httpx.URL:
    url = httpx.URL(value)
    if (url.scheme not in {"http", "https"} or not url.host or url.userinfo
            or url.raw_path != b"/" or "?" in value or "#" in value or "@" in value):
        raise ValueError("device origin must be an HTTP(S) origin")
    return url


def simulator_origin(environ: Mapping[str, str]) -> str:
    """Return the authorized simulator origin, or raise a policy error.

    Simulator reads are opt-in and hard-pinned: the operator must set
    ``PRINTDIAG_ALLOW_SIMULATOR=1`` and the origin must be exactly the loopback
    development simulator. Physical printers are never authorized by this mode.
    """
    if environ.get("PRINTDIAG_ALLOW_SIMULATOR") != "1":
        raise ValueError("simulator mode requires explicit PRINTDIAG_ALLOW_SIMULATOR=1")
    origin = environ.get("PRINTDIAG_SIMULATOR_URL", AUTHORIZED_SIMULATOR_ORIGIN)
    if origin != AUTHORIZED_SIMULATOR_ORIGIN:
        raise ValueError(f"only {AUTHORIZED_SIMULATOR_ORIGIN} is authorized for simulator reads")
    return origin


class ReadOnlyTransport(httpx.BaseTransport):
    def __init__(self, origin: httpx.URL, inner: httpx.BaseTransport, deadline: float,
                 *, clock: Callable[[], float] = monotonic, cancelled: Event | None = None) -> None:
        self._origin = origin
        self._inner = inner
        self._deadline = deadline
        self._clock = clock
        self._cancelled = cancelled

    def _remaining(self) -> float:
        remaining = self._deadline - self._clock()
        if remaining <= 0 or (self._cancelled is not None and self._cancelled.is_set()):
            raise MarlinTimeoutError("device deadline expired")
        return remaining

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        try:
            has_body = bool(request.content)
        except httpx.RequestNotRead:
            has_body = True
        if (request.method != "GET" or url.raw_path not in GET_PATHS or url.fragment or url.userinfo
                or (url.scheme, url.netloc) != (self._origin.scheme, self._origin.netloc)
                or request.headers.get_list("host") != [self._origin.netloc.decode("ascii")]
                or has_body or "transfer-encoding" in request.headers
                or request.headers.get_list("content-length") not in ([], ["0"])):
            raise AccessDenied("request violates the device read policy")
        remaining = self._remaining()
        timeout = request.extensions.get("timeout", {})
        request.extensions["timeout"] = {
            key: min(timeout.get(key) or remaining, remaining) for key in ("connect", "read", "write", "pool")
        }
        request.headers["accept-encoding"] = "identity"
        response = self._inner.handle_request(request)
        try:
            self._remaining()
            if 300 <= response.status_code < 400:
                raise AccessDenied("device redirects are forbidden")
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise MarlinProtocolError("encoded device responses are forbidden")
            body = bytearray()
            chunks = (response.content,) if response.is_stream_consumed else response.iter_raw()
            for chunk in chunks:
                self._remaining()
                if len(body) + len(chunk) > MAX_BODY_BYTES:
                    raise MarlinProtocolError("device response exceeds size limit")
                body.extend(chunk)
            return httpx.Response(response.status_code, content=bytes(body))
        finally:
            response.close()

    def close(self) -> None:
        self._inner.close()


def synthetic_device(request: httpx.Request) -> httpx.Response:
    """Stationary synthetic sample, independent of historical experiments."""
    if request.url.path == "/state":
        position = {"x": 12.0, "y": 15.0, "z": 8.0}
        return httpx.Response(200, json={
            "kernel": {"sim_seconds": 4.0, "timers_active": True, "watchdog": {"resets": 0}},
            "thermal_faults": {"faults": []},
            "components": {"Cartesian Kinematic System": {"state": {
                "position": position, "effectors": [{"stepper_position": dict.fromkeys(position, 0)}],
            }}},
        })
    if request.url.path == "/idle":
        return httpx.Response(200, json={"idle": True, "moves_queued": 0})
    return httpx.Response(200, json={"identifier": "EndStop", "state": {
        "triggered": request.url.path.endswith(ENDSTOPS["x_min"]), "geometric_triggered": False, "enabled": True,
    }})


def _object(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value: object) -> bool:
    try:
        return type(value) in (int, float) and isfinite(value)
    except OverflowError:
        return False


def _count(value: object) -> bool:
    return type(value) is int and 0 <= value <= 2**63 - 1


def _xyz(value: object) -> XYZ | None:
    data = _object(value)
    if not all(_number(data.get(axis)) for axis in ("x", "y", "z")):
        return None
    return XYZ(x=data["x"], y=data["y"], z=data["z"])


def _field(value, valid: Callable, name: str, issues: list[Issue]):
    if valid(value):
        return value
    issues.append(Issue(code="FIELD_UNAVAILABLE" if value is None else "INVALID_DEVICE_RESPONSE", field=name))
    return None


def _optional_field(value, valid: Callable, name: str, issues: list[Issue]):
    """Optional projection: absent is fine (unavailable), malformed is an issue."""
    if value is None:
        return None
    if valid(value):
        if type(value) is str:
            return value.strip() or None
        return value
    issues.append(Issue(code="INVALID_DEVICE_RESPONSE", field=name))
    return None


def _heater_projection(prefix: str, component: object, issues: list[Issue]) -> dict[str, object]:
    """Project one heater component; target stays None when the build omits it."""
    state = _object(_object(component).get("state"))
    target = state.get("target_temperature")
    if target is None:
        target = state.get("target")
    return {
        f"{prefix}_temperature": _optional_field(state.get("temperature"), _number, f"{prefix}.temperature", issues),
        f"{prefix}_target_temperature": _optional_field(target, _number, f"{prefix}.target_temperature", issues),
        f"{prefix}_ambient_temperature": _optional_field(
            state.get("ambient_temperature"), _number, f"{prefix}.ambient_temperature", issues),
        f"{prefix}_pwm_duty": _optional_field(state.get("pwm_duty"), _number, f"{prefix}.pwm_duty", issues),
        f"{prefix}_element_ok": _optional_field(state.get("element_ok"), _is_bool, f"{prefix}.element_ok", issues),
        f"{prefix}_sensor_mode": _optional_field(
            state.get("sensor_mode"), lambda value: isinstance(value, str) and 0 < len(value.strip()) <= 40,
            f"{prefix}.sensor_mode", issues),
        f"{prefix}_sensor_error": _optional_field(
            state.get("sensor_error"), _number, f"{prefix}.sensor_error", issues),
    }


def _is_bool(value: object) -> bool:
    return type(value) is bool


def _code(error: MarlinError) -> ErrorCode:
    match error:
        case AccessDenied():
            return "ACCESS_DENIED"
        case MarlinTimeoutError():
            return "DEVICE_TIMEOUT"
        case MarlinHTTPError():
            return "DEVICE_HTTP_ERROR"
        case MarlinProtocolError():
            return "INVALID_DEVICE_RESPONSE"
        case _:
            return "DEVICE_UNAVAILABLE"


class DeviceReads:
    """Own a bounded client per tool call. Only projection methods are exposed."""

    def __init__(self, *, live_origin: str | None = None, http_timeout_s: float = 1.0,
                 freshness_s: float = 1.0, clock: Callable[[], float] = monotonic,
                 mock_handler: Callable[[httpx.Request], httpx.Response] = synthetic_device,
                 provenance: DeviceProvenance | None = None) -> None:
        if live_origin is None:
            self.source = "mock"
            self.provenance: DeviceProvenance = "mock"
        else:
            self.source = "live_read"
            # Only the local simulator is authorized in this stage; a future
            # physical-printer mode must declare its own provenance explicitly.
            self.provenance = provenance or "simulator"
        self._origin = device_origin(live_origin if live_origin is not None else MOCK_ORIGIN)
        if live_origin is not None and (
            self._origin.host not in {"127.0.0.1", "::1"} or self._origin.port in {8099, 8200, 8299}
        ):
            raise ValueError("live Agent reads require a loopback development origin")
        if not all(isfinite(value) and value > 0 for value in (http_timeout_s, freshness_s)):
            raise ValueError("device budgets must be finite and positive")
        self._config = Config(str(self._origin), http_timeout_s, "INFO")
        self._freshness_s = freshness_s
        self._clock = clock
        self._mock_handler = mock_handler

    def _http_client(self, deadline: float, cancelled: Event | None) -> httpx.Client:
        inner = (httpx.MockTransport(self._mock_handler) if self.source == "mock"
                 else httpx.HTTPTransport(trust_env=False, retries=0))
        gate = ReadOnlyTransport(self._origin, inner, deadline, clock=self._clock, cancelled=cancelled)
        return httpx.Client(transport=gate, trust_env=False, follow_redirects=False)

    def _read(self, requests, deadline: float, cancelled: Event | None) -> tuple[dict[str, dict], list[Issue]]:
        started = {}
        payloads = {}
        issues = []
        with self._http_client(deadline, cancelled) as http:
            client = MarlinClient(http, config=self._config)
            for field, method, arguments in requests:
                started[field] = self._clock()
                try:
                    payloads[field] = method(client, *arguments)
                except MarlinError as exc:
                    code = _code(exc)
                    issues.append(Issue(code=code, field=field, retryable=code in {
                        "DEVICE_UNAVAILABLE", "DEVICE_TIMEOUT", "DEVICE_HTTP_ERROR",
                    }))
        finished = self._clock()
        for field in list(payloads):
            if finished - started[field] > self._freshness_s:
                payloads.pop(field)
                issues.append(Issue(code="STALE_OBSERVATION", field=field, retryable=True))
        return payloads, issues

    def printer_state(self, deadline: float, *, cancelled: Event | None = None
                      ) -> tuple[PrinterStateProjection | None, list[Issue]]:
        payloads, issues = self._read([
            ("state", MarlinClient.get_state, ()), ("idle", MarlinClient.get_idle_status, ()),
        ], deadline, cancelled)
        data = dict.fromkeys(PrinterStateProjection.model_fields)
        idle = payloads.get("idle")
        if idle is not None:
            data["idle"] = idle["idle"]
            data["moves_queued"] = _field(idle.get("moves_queued"), _count, "moves_queued", issues)
        state = payloads.get("state")
        if state is not None:
            kernel = state["kernel"]
            sim_time = kernel.get("sim_seconds")
            idle_time = idle.get("sim_seconds") if idle is not None else None
            if _number(sim_time) and _number(idle_time) and abs(sim_time - idle_time) > self._freshness_s:
                issues.append(Issue(code="STALE_OBSERVATION", field="state", retryable=True))
            else:
                data["sim_seconds"] = _field(sim_time, lambda value: _number(value) and value >= 0,
                                             "sim_seconds", issues)
                data["timers_active"] = _field(kernel.get("timers_active"), _is_bool, "timers_active", issues)
                data["watchdog_resets"] = _field(_object(kernel.get("watchdog")).get("resets"),
                                                 _count, "watchdog_resets", issues)
                faults = _field(_object(state.get("thermal_faults")).get("faults"),
                                lambda value: isinstance(value, list) and all(isinstance(item, dict) for item in value),
                                "thermal_fault_count", issues)
                data["thermal_fault_count"] = len(faults) if faults is not None else None
                kin = _object(_object(state["components"].get("Cartesian Kinematic System")).get("state"))
                effectors = kin.get("effectors")
                effector = _object(effectors[0]) if isinstance(effectors, list) and len(effectors) == 1 else {}
                for field, value in (("position_mm", kin.get("position")),
                                     ("stepper_position", effector.get("stepper_position"))):
                    data[field] = _xyz(value)
                    if data[field] is None:
                        issues.append(Issue(code="FIELD_UNAVAILABLE" if value is None else
                                            "INVALID_DEVICE_RESPONSE", field=field))
                components = state["components"]
                for prefix, component_name in HEATERS.items():
                    data.update(_heater_projection(prefix, components.get(component_name), issues))
        return (PrinterStateProjection(**data) if any(value is not None for value in data.values()) else None), issues

    def endstops(self, axes: list[Axis], deadline: float, *, cancelled: Event | None = None
                 ) -> tuple[EndstopData, list[Issue]]:
        payloads, issues = self._read(
            [(axis, MarlinClient.get_component_state, (ENDSTOPS[axis],)) for axis in axes], deadline, cancelled,
        )
        components = []
        for axis in axes:
            payload = payloads.get(axis)
            if payload is not None and payload["identifier"] != "EndStop":
                issues.append(Issue(code="INVALID_DEVICE_RESPONSE", field=axis))
                payload = None
            values = dict.fromkeys(("triggered", "geometric_triggered", "enabled"))
            if payload is not None:
                for field in values:
                    values[field] = _field(payload["state"].get(field), _is_bool, f"{axis}.{field}", issues)
            components.append(EndstopComponentProjection(axis=axis, **values))
        return EndstopData(components=components), issues


def simulator_device(*, http_timeout_s: float = 1.0, freshness_s: float = 1.0,
                     clock: Callable[[], float] = monotonic) -> DeviceReads:
    """Build the explicitly authorized, read-only local Marlin Simulator device."""
    return DeviceReads(
        live_origin=AUTHORIZED_SIMULATOR_ORIGIN, http_timeout_s=http_timeout_s,
        freshness_s=freshness_s, clock=clock, provenance="simulator",
    )
