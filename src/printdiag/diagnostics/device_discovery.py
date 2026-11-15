"""Read component capabilities without assuming a common device state schema."""

from collections.abc import Callable, Iterable
from math import isfinite
from threading import Event
from typing import Any

from printdiag.clients.marlin_client import MarlinClient, MarlinError, MarlinHTTPError

JsonObject = dict[str, Any]


def error_details(exc: Exception) -> JsonObject:
    details = {"type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, MarlinHTTPError):
        details["http_status"] = exc.status_code
    return details


def read_json(read: Callable[[], Any]) -> JsonObject:
    try:
        return {"status": "ok", "raw": read(), "error": None}
    except (MarlinError, ValueError) as exc:
        return {"status": "error", "raw": None, "error": error_details(exc)}


def _normalize_component(raw: JsonObject) -> JsonObject | None:
    identifier = raw["identifier"]
    state = raw["state"]
    if identifier == "Heater":
        temperature = state.get("temperature")
        if type(temperature) in (int, float) and isfinite(temperature):
            return {"kind": "heater", "temperature_c": temperature}
    if identifier == "EndStop":
        triggered = state.get("triggered")
        if isinstance(triggered, bool):
            return {"kind": "endstop", "triggered": triggered}
    return None


def probe_components(
    client: MarlinClient, names: Iterable[str], stop_event: Event
) -> list[JsonObject]:
    records = []
    for name in names:
        if stop_event.is_set():
            break
        record = read_json(lambda: client.get_component_state(name))
        raw = record["raw"]
        records.append({
            "name": name,
            **record,
            "readable_fields": sorted(raw["state"]) if raw is not None else [],
            "normalized": _normalize_component(raw) if raw is not None else None,
        })
    return records
