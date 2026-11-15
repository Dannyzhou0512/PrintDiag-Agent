"""Collect bounded read-only samples through the existing Marlin client."""

import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from threading import TIMEOUT_MAX, Event
from uuid import uuid4

from printdiag.clients.marlin_client import (
    MarlinBusyError,
    MarlinClient,
    MarlinError,
    QueryResult,
    QueryStatus,
    reserve_query_channel,
)
from printdiag.diagnostics.device_discovery import (
    JsonObject,
    error_details,
    probe_components,
    read_json,
)

READ_ONLY_QUERIES = ("M115", "M105", "M114", "M119")
SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _validate_series(count: int, interval_s: float) -> None:
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(interval_s) not in (int, float) or not 0 <= interval_s <= TIMEOUT_MAX:
        raise ValueError(f"interval_s must be a number from 0 to {TIMEOUT_MAX:g}")


def _validate_query_timeout(timeout_s: float) -> None:
    if type(timeout_s) not in (int, float) or not isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("query timeout must be a finite, positive number")


def _query_record(result: QueryResult, error: JsonObject | None = None) -> JsonObject:
    return {
        **asdict(result),
        "status": result.status.value,
        "response": list(result.response),
        "error": error,
    }


def _query_error(command: str, exc: MarlinError, elapsed_s: float | None) -> JsonObject:
    if exc.partial_result is not None:
        return _query_record(exc.partial_result, error_details(exc))
    return {
        "command": command,
        "submitted": None,
        "idle": None,
        "status": "error",
        "response": None,
        "serial_from": None,
        "serial_pages": [],
        "firmware_name": None,
        "elapsed_s": elapsed_s,
        "error": error_details(exc),
    }


@dataclass(frozen=True)
class BaselineBatch:
    manifest: JsonObject
    discovery: JsonObject
    snapshots: list[JsonObject]
    queries: list[JsonObject]


class StateCollector:
    """Collect with cooperative cancellation between requests and during delays."""

    def __init__(self, client: MarlinClient, *, stop_event: Event | None = None) -> None:
        self._client = client
        self._stop = stop_event if stop_event is not None else Event()

    def collect_once(self, sequence: int = 1) -> JsonObject:
        sampled_at = _utc_now()
        reading = (
            {"status": "cancelled", "raw": None, "error": None}
            if self._stop.is_set()
            else read_json(self._client.get_state)
        )
        return {"sequence": sequence, "sampled_at_utc": sampled_at, **reading}

    def collect_series(self, count: int, interval_s: float) -> list[JsonObject]:
        _validate_series(count, interval_s)
        snapshots = []
        for sequence in range(1, count + 1):
            if self._stop.is_set():
                break
            started = time.monotonic()
            snapshots.append(self.collect_once(sequence))
            if sequence < count:
                delay = max(0.0, interval_s - (time.monotonic() - started))
                if self._stop.wait(delay):
                    break
        return snapshots

    def collect_component_states(self, names: Iterable[str]) -> list[JsonObject]:
        return probe_components(self._client, names, self._stop)

    def discover_device(self) -> JsonObject:
        inventory = (
            {"status": "cancelled", "raw": None, "error": None}
            if self._stop.is_set()
            else read_json(self._client.list_components)
        )
        initial_state = self.collect_once(sequence=0)
        names = inventory["raw"] if inventory["raw"] is not None else []
        components = self.collect_component_states(names)
        return {
            "inventory": inventory,
            "initial_state": initial_state,
            "components": components,
            "unprobed_names": names[len(components):],
        }

    def collect_firmware_queries(
        self,
        commands: Iterable[str] = READ_ONLY_QUERIES,
        *,
        timeout_s: float = 10.0,
    ) -> list[JsonObject]:
        _validate_query_timeout(timeout_s)
        if self._stop.is_set():
            return []
        results = []
        try:
            with reserve_query_channel():
                for command in commands:
                    if self._stop.is_set():
                        break
                    started = time.monotonic()
                    try:
                        result = self._client.execute_query(command, timeout_s=timeout_s)
                    except MarlinError as exc:
                        results.append(_query_error(command, exc, time.monotonic() - started))
                    else:
                        results.append(_query_record(result))
        except MarlinBusyError as exc:
            return [_query_error(command, exc, None) for command in commands]
        return results

    def collect_baseline(
        self,
        count: int,
        interval_s: float,
        *,
        with_queries: bool = False,
        query_timeout_s: float = 10.0,
    ) -> BaselineBatch:
        _validate_series(count, interval_s)
        _validate_query_timeout(query_timeout_s)
        batch_id = uuid4().hex
        started_at = _utc_now()
        discovery = self.discover_device()
        snapshots = self.collect_series(count, interval_s)
        commands = READ_ONLY_QUERIES if with_queries else ()
        queries = self.collect_firmware_queries(commands, timeout_s=query_timeout_s) if commands else []

        inventory = discovery["inventory"]
        initial_state = discovery["initial_state"]
        components = discovery["components"]
        component_ok = sum(row["status"] == "ok" for row in components)
        snapshot_ok = sum(row["status"] == "ok" for row in snapshots)
        confirmed = [row for row in queries if row["status"] == QueryStatus.CONFIRMED]
        name_mismatches = []
        duplicate_names = []
        if inventory["status"] == "ok":
            names = inventory["raw"]
            expected = set(names)
            duplicate_names = [name for name, count in Counter(names).items() if count > 1]
            for row in [initial_state, *snapshots]:
                if row["status"] != "ok":
                    continue
                actual = set(row["raw"]["components"])
                if actual != expected:
                    name_mismatches.append({
                        "sequence": row["sequence"],
                        "missing": sorted(expected - actual),
                        "extra": sorted(actual - expected),
                    })
        complete = (
            inventory["status"] == initial_state["status"] == "ok"
            and component_ok == len(inventory["raw"])
            and snapshot_ok == count
            and len(confirmed) == len(commands)
            and not name_mismatches
            and not duplicate_names
        )
        status = "complete"
        if self._stop.is_set():
            status = "cancelled"
        elif not complete:
            has_data = (
                inventory["status"] == "ok" or initial_state["status"] == "ok"
                or snapshot_ok or confirmed
            )
            status = "partial" if has_data else "failed"

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "batch_id": batch_id,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "status": status,
            "parameters": {
                "samples": count,
                "interval_s": interval_s,
                "with_queries": with_queries,
                "query_timeout_s": query_timeout_s,
            },
            "firmware": next(
                (row["firmware_name"] for row in confirmed if row["command"] == "M115"),
                None,
            ),
            "integrity": {
                "inventory_status": inventory["status"],
                "initial_state_status": initial_state["status"],
                "components_discovered": (
                    len(inventory["raw"]) if inventory["status"] == "ok" else None
                ),
                "components_probed": len(components),
                "components_ok": component_ok,
                "duplicate_component_names": duplicate_names,
                "component_name_mismatches": name_mismatches,
                "empty_component_states": [
                    row["name"] for row in components
                    if row["status"] == "ok" and not row["readable_fields"]
                ],
                "snapshots_expected": count,
                "snapshots_recorded": len(snapshots),
                "snapshots_ok": snapshot_ok,
                "queries_expected": len(commands),
                "queries_recorded": len(queries),
                "queries_confirmed": len(confirmed),
            },
        }
        return BaselineBatch(manifest, discovery, snapshots, queries)
