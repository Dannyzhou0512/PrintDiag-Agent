"""Synchronous, read-only client for the Marlin Simulator HTTP API."""

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from threading import Lock, RLock
from typing import Any, Self
from urllib.parse import quote, urlsplit

import httpx

from printdiag.config import Config, load_config


class MarlinError(Exception):
    """Base exception for Marlin client failures."""

    partial_result: "QueryResult | None" = None


class MarlinConnectionError(MarlinError):
    """The HTTP transport failed."""


class MarlinTimeoutError(MarlinError):
    """An HTTP request timed out."""


class MarlinDeadlineError(MarlinTimeoutError):
    """The polling deadline expired."""


class MarlinHTTPError(MarlinError):
    def __init__(self, method: str, path: str, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"{method} {path} returned HTTP {status_code}")


class MarlinProtocolError(MarlinError):
    """The simulator returned malformed or inconsistent response fields."""


class MarlinCommandError(MarlinError):
    """The command is unauthorized or was rejected by the simulator."""


class MarlinBusyError(MarlinError):
    """Another G-code operation owns the process-wide query channel."""


_QUERY_CHANNEL = RLock()
_ACTIVE_COMMAND = Lock()


@contextmanager
def reserve_query_channel() -> Iterator[None]:
    """Reserve the process-wide channel for a sequence of read-only queries."""
    if not _QUERY_CHANNEL.acquire(blocking=False):
        raise MarlinBusyError("Another G-code operation is in progress")
    try:
        yield
    finally:
        _QUERY_CHANNEL.release()


@dataclass(frozen=True)
class GcodeSubmission:
    command: str
    serial_from: int


@dataclass(frozen=True)
class SerialLine:
    seq: int
    stream: int
    text: str
    sim_seconds: float


@dataclass(frozen=True)
class SerialPage:
    lines: tuple[SerialLine, ...]
    next_seq: int
    head_seq: int
    oldest_seq: int
    dropped: int
    max_lines: int


class QueryStatus(StrEnum):
    CONFIRMED = "confirmed"
    IDLE_TIMEOUT = "idle_timeout"
    SERIAL_TIMEOUT = "serial_timeout"
    SERIAL_GAP = "serial_gap"
    MISSING_RESPONSE = "missing_response"
    FIRMWARE_ERROR = "firmware_error"
    ERROR = "error"


@dataclass(frozen=True)
class SerialCursorPage:
    since: int
    next_seq: int
    head_seq: int
    oldest_seq: int
    max_lines: int
    lines: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class QueryResult:
    command: str
    submitted: bool
    idle: bool
    status: QueryStatus
    response: tuple[str, ...]
    elapsed_s: float
    serial_from: int
    firmware_name: str | None
    serial_pages: tuple[SerialCursorPage, ...] = ()


_RESPONSE_MARKERS = {
    "M105": re.compile(r"^ok\s+T(?:\d+)?:\s*[+-]?\d"),
    "M114": re.compile(r"^X:\s*[+-]?\d.*\sY:\s*[+-]?\d.*\sZ:\s*[+-]?\d"),
    "M115": re.compile(
        r"^FIRMWARE_NAME:\s*((?![A-Z][A-Z0-9_]*:)\S.*?)(?=\s+[A-Z][A-Z0-9_]*:|$)"
    ),
    "M119": re.compile(r"^Reporting endstop status\b"),
}
_FIRMWARE_ERROR = re.compile(r'^(?:echo:\s*)?(?:Error:|Unknown command\b)', re.I)
_ENDSTOP_STATUS = re.compile(r"^[a-z0-9_]+:\s+(?:open|TRIGGERED)$", re.I)
_POLL_INTERVAL_S = 0.05
_FIRMWARE_STREAM = 3


def _positive_timeout(value: float, name: str) -> float:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than 0")
    return value


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise MarlinDeadlineError("Simulator polling deadline expired")
    return remaining


def _is_sequence(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_timestamp(value: object) -> bool:
    return type(value) in {int, float} and 0 <= value < float("inf")


class MarlinClient:
    """Use project settings and leave an injected HTTP client open for its owner."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        http_timeout_s: float | None = None,
        config: Config | None = None,
    ) -> None:
        config = config if config is not None else load_config()
        try:
            urlsplit(config.marlin_base_url)
            url = httpx.URL(config.marlin_base_url)
        except (ValueError, httpx.InvalidURL):
            raise ValueError("MARLIN_BASE_URL must be a valid HTTP or HTTPS base URL") from None
        if (
            url.scheme not in {"http", "https"} or not url.host or url.userinfo
            or "?" in config.marlin_base_url or "#" in config.marlin_base_url
        ):
            raise ValueError("MARLIN_BASE_URL must be HTTP or HTTPS without credentials, query or fragment")
        self._base_url = str(url).rstrip("/")
        self._http_timeout = _positive_timeout(
            config.marlin_http_timeout if http_timeout_s is None else http_timeout_s,
            "http_timeout_s",
        )
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(trust_env=False)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        deadline: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        timeout = self._http_timeout
        if deadline is not None:
            timeout = min(timeout, _remaining(deadline))
        try:
            response = self._client.request(
                method, self._base_url + path, timeout=timeout,
                follow_redirects=False, **kwargs,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise MarlinHTTPError(method, path, exc.response.status_code) from None
        except httpx.TimeoutException:
            raise MarlinTimeoutError(f"{method} {path} timed out") from None
        except httpx.RequestError:
            raise MarlinConnectionError(
                f"{method} {path} failed to communicate with Marlin Simulator"
            ) from None

        try:
            payload = response.json()
        except (ValueError, RecursionError):
            raise MarlinProtocolError(f"{method} {path} returned invalid JSON") from None
        if not isinstance(payload, dict):
            raise MarlinProtocolError(f"{method} {path} must return a JSON object")
        return payload

    def ping(self) -> bool:
        payload = self._request("GET", "/ping")
        if payload.get("ok") is not True:
            raise MarlinProtocolError("GET /ping must return ok=true")
        return True

    def list_components(self) -> list[str]:
        names = self._request("GET", "/components").get("components")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise MarlinProtocolError("GET /components must return a list of component names")
        return names

    def get_state(self) -> dict[str, Any]:
        payload = self._request("GET", "/state")
        if not isinstance(payload.get("kernel"), dict) or not isinstance(payload.get("components"), dict):
            raise MarlinProtocolError("GET /state must return kernel and components objects")
        return payload

    def get_component_state(self, name: str) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip() or name in {".", ".."}:
            raise ValueError("A component name is required")
        payload = self._request("GET", f"/state/{quote(name, safe='')}")
        if not isinstance(payload.get("identifier"), str) or not isinstance(payload.get("state"), dict):
            raise MarlinProtocolError("Component response must contain an identifier and a state object")
        return payload

    def get_idle_status(self) -> dict[str, Any]:
        return self._idle_status()

    def _idle_status(self, deadline: float | None = None) -> dict[str, Any]:
        payload = self._request("GET", "/idle", deadline=deadline)
        if not isinstance(payload.get("idle"), bool):
            raise MarlinProtocolError("GET /idle must return a boolean idle field")
        return payload

    def get_serial_output(
        self, since: int = 0, stream: int | None = None, limit: int = 200
    ) -> SerialPage:
        if not _is_sequence(since):
            raise ValueError("since must be a non-negative integer")
        if stream is not None and (not _is_sequence(stream) or stream > 3):
            raise ValueError("stream must be None or an integer from 0 to 3")
        if type(limit) is not int or not 1 <= limit <= 5000:
            raise ValueError("limit must be an integer from 1 to 5000")
        return self._serial_page(since, stream, limit)

    def _serial_page(
        self, since: int, stream: int | None, limit: int, deadline: float | None = None
    ) -> SerialPage:
        params = {"since": since, "limit": limit}
        if stream is not None:
            params["stream"] = stream
        payload = self._request("GET", "/serial", params=params, deadline=deadline)
        fields = ("next_seq", "head_seq", "oldest_seq", "dropped", "max_lines")
        for field in fields:
            if not _is_sequence(payload.get(field)):
                raise MarlinProtocolError(f"GET /serial returned an invalid {field}")

        raw_lines = payload.get("lines")
        if not isinstance(raw_lines, list) or len(raw_lines) > limit:
            raise MarlinProtocolError("GET /serial returned an invalid lines array")
        lines = []
        next_seq = since
        for row in raw_lines:
            if (
                not isinstance(row, dict)
                or not _is_sequence(row.get("seq"))
                or row["seq"] < next_seq
                or not _is_sequence(row.get("stream"))
                or row["stream"] > 3
                or (stream is not None and row["stream"] != stream)
                or not isinstance(row.get("text"), str)
                or not _is_timestamp(row.get("sim_seconds"))
            ):
                raise MarlinProtocolError("GET /serial returned an invalid serial line")
            lines.append(SerialLine(row["seq"], row["stream"], row["text"], row["sim_seconds"]))
            next_seq = row["seq"] + 1
        if payload["next_seq"] != next_seq:
            raise MarlinProtocolError("GET /serial returned an inconsistent next_seq")
        return SerialPage(tuple(lines), **{field: payload[field] for field in fields})

    @contextmanager
    def _command_slot(self) -> Iterator[None]:
        with reserve_query_channel():
            if not _ACTIVE_COMMAND.acquire(blocking=False):
                raise MarlinBusyError("Another G-code operation is in progress")
            try:
                yield
            finally:
                _ACTIVE_COMMAND.release()

    def send_gcode(self, command: str) -> GcodeSubmission:
        """Submit an allowlisted command and return its serial cursor."""
        with self._command_slot():
            return self._submit(command)

    def _submit(self, command: str, deadline: float | None = None) -> GcodeSubmission:
        if type(command) is not str or command not in _RESPONSE_MARKERS:
            raise MarlinCommandError("Only the exact read-only commands M105, M114, M115 and M119 are allowed")
        payload = self._request("POST", "/gcode", json={"command": command}, deadline=deadline)
        if payload.get("ok") is False:
            raise MarlinCommandError("Marlin Simulator rejected the G-code submission")
        if (
            payload.get("ok") is not True
            or payload.get("command") != command
            or not _is_sequence(payload.get("serial_from"))
        ):
            raise MarlinProtocolError("POST /gcode returned an invalid submission receipt")
        return GcodeSubmission(command, payload["serial_from"])

    def wait_until_idle(self, timeout_s: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + _positive_timeout(timeout_s, "timeout_s")
        return self._wait_for_idle(deadline)

    def _wait_for_idle(self, deadline: float) -> dict[str, Any]:
        while True:
            payload = self._idle_status(deadline)
            remaining = _remaining(deadline)
            if payload["idle"]:
                return payload
            time.sleep(min(_POLL_INTERVAL_S, remaining))

    def execute_query(self, command: str, timeout_s: float = 10.0) -> QueryResult:
        """Confirm a query from its firmware data and terminal acknowledgement."""
        start = time.monotonic()
        deadline = start + _positive_timeout(timeout_s, "timeout_s")
        lines: list[str] = []
        pages: list[SerialCursorPage] = []
        firmware_name = None
        idle = False
        with self._command_slot():
            submission = self._submit(command, deadline)
            try:
                self._wait_for_idle(deadline)
                idle = True
                status, firmware_name = self._collect_response(
                    submission, deadline, lines, pages
                )
            except MarlinDeadlineError:
                status = QueryStatus.SERIAL_TIMEOUT if idle else QueryStatus.IDLE_TIMEOUT
            except MarlinError as exc:
                exc.partial_result = QueryResult(
                    command, True, idle, QueryStatus.ERROR, tuple(lines),
                    time.monotonic() - start, submission.serial_from, None, tuple(pages),
                )
                raise
        return QueryResult(
            command, True, idle, status, tuple(lines), time.monotonic() - start,
            submission.serial_from, firmware_name, tuple(pages),
        )

    def _collect_response(
        self, submission: GcodeSubmission, deadline: float, response: list[str],
        pages: list[SerialCursorPage],
    ) -> tuple[QueryStatus, str | None]:
        cursor = submission.serial_from
        marker = _RESPONSE_MARKERS[submission.command]
        matched = False
        endstop_status_seen = False
        firmware_name = None
        while True:
            # Sequences are global across streams. Read all streams for continuity.
            page = self._serial_page(cursor, None, 200, deadline)
            pages.append(SerialCursorPage(
                cursor, page.next_seq, page.head_seq, page.oldest_seq, page.max_lines,
                tuple((line.seq, line.stream) for line in page.lines),
            ))
            _remaining(deadline)
            if not page.max_lines or not page.oldest_seq <= cursor <= page.head_seq:
                return QueryStatus.SERIAL_GAP, firmware_name
            if page.next_seq > page.head_seq:
                return QueryStatus.SERIAL_GAP, firmware_name

            for line in page.lines:
                if line.seq != cursor:
                    return QueryStatus.SERIAL_GAP, firmware_name
                cursor += 1
                if line.stream != _FIRMWARE_STREAM:
                    continue
                response.append(line.text)
                text = line.text.strip()
                if _FIRMWARE_ERROR.match(text):
                    return QueryStatus.FIRMWARE_ERROR, firmware_name
                evidence = marker.match(text)
                if evidence:
                    matched = True
                    if submission.command == "M115":
                        firmware_name = evidence[1].strip()
                if submission.command == "M119" and matched and _ENDSTOP_STATUS.fullmatch(text):
                    endstop_status_seen = True
                if text.split(maxsplit=1)[:1] == ["ok"]:
                    confirmed = matched and (submission.command != "M119" or endstop_status_seen)
                    status = QueryStatus.CONFIRMED if confirmed else QueryStatus.MISSING_RESPONSE
                    return status, firmware_name

            if page.next_seq == page.head_seq:
                return QueryStatus.MISSING_RESPONSE, firmware_name
            if not page.lines:
                return QueryStatus.SERIAL_GAP, firmware_name
            cursor = page.next_seq
