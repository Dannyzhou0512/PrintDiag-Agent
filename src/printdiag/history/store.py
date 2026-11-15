"""SQLite storage for validated diagnosis history.

One short-lived connection per operation keeps the implementation thread-safe
under FastAPI's threadpool. SQLite WAL plus busy_timeout prevents partial
writes and transient lock errors for this single-user local deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

SCHEMA_VERSION = 1
HISTORY_SCHEMA_VERSION = SCHEMA_VERSION


class HistoryRecordError(ValueError):
    """Raised when a record cannot be stored without inventing data."""


@dataclass(frozen=True)
class SaveResult:
    created: bool
    diagnosis_id: str
    request_id: str


_JSON_COLUMNS = {
    "validated_report", "evidence_summary", "rag_references", "v03_validation",
    "v2_codes", "safety_result", "device_snapshot_reference", "answer",
}

_COLUMNS = (
    "diagnosis_id", "request_id", "created_at", "completed_at", "user_question",
    "intent", "source_mode", "model_name", "model_variant", "diagnosis_status",
    "fault_category", "validated_report", "evidence_summary", "rag_references",
    "v03_validation", "v2_decision", "v2_codes", "safety_result",
    "device_source", "device_snapshot_reference", "answer", "duration_ms",
    "error_code", "schema_version", "persistence_status",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class HistoryStore:
    """Small, synchronous SQLite repository with an explicit schema version."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise HistoryRecordError(
                    f"unsupported history schema {version}; expected {SCHEMA_VERSION}")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnoses (
                    diagnosis_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    user_question TEXT NOT NULL,
                    intent TEXT,
                    source_mode TEXT NOT NULL,
                    model_name TEXT,
                    model_variant TEXT,
                    diagnosis_status TEXT NOT NULL,
                    fault_category TEXT,
                    validated_report TEXT,
                    evidence_summary TEXT,
                    rag_references TEXT,
                    v03_validation TEXT,
                    v2_decision TEXT,
                    v2_codes TEXT,
                    safety_result TEXT,
                    device_source TEXT,
                    device_snapshot_reference TEXT,
                    answer TEXT,
                    duration_ms REAL,
                    error_code TEXT,
                    schema_version INTEGER NOT NULL,
                    persistence_status TEXT NOT NULL DEFAULT 'saved'
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_diagnoses_created "
                "ON diagnoses(created_at DESC)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_diagnoses_status "
                "ON diagnoses(diagnosis_status, created_at DESC)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_diagnoses_source "
                "ON diagnoses(source_mode, created_at DESC)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_diagnoses_model "
                "ON diagnoses(model_variant, created_at DESC)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS history_meta ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "INSERT INTO history_meta(key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value "
                "WHERE history_meta.value <> excluded.value", (str(SCHEMA_VERSION),))
            if version != SCHEMA_VERSION:
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def save(self, record: dict[str, Any]) -> SaveResult:
        """Insert once by request_id.

        Duplicate submissions return the existing row and never create a
        second diagnosis. The transaction makes the duplicate check and insert
        atomic.
        """
        required = ("diagnosis_id", "request_id", "user_question", "source_mode",
                    "diagnosis_status")
        for key in required:
            if not record.get(key):
                raise HistoryRecordError(f"missing required history field: {key}")
        payload = {column: record.get(column) for column in _COLUMNS}
        payload["schema_version"] = SCHEMA_VERSION
        payload["persistence_status"] = record.get("persistence_status") or "saved"
        for column in _JSON_COLUMNS:
            if column in payload:
                payload[column] = _json(payload[column])
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT diagnosis_id, request_id FROM diagnoses WHERE request_id = ?",
                (payload["request_id"],)).fetchone()
            if existing is not None:
                connection.commit()
                return SaveResult(False, existing["diagnosis_id"], existing["request_id"])
            placeholders = ", ".join("?" for _ in _COLUMNS)
            connection.execute(
                f"INSERT INTO diagnoses ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                tuple(payload.get(column) for column in _COLUMNS))
            connection.commit()
        return SaveResult(True, payload["diagnosis_id"], payload["request_id"])

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for column in _JSON_COLUMNS:
            value = result.get(column)
            if isinstance(value, str) and value:
                try:
                    result[column] = json.loads(value)
                except ValueError:
                    # Keep malformed historical JSON visible as text rather than
                    # silently dropping it.
                    pass
        return result

    def get(self, diagnosis_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM diagnoses WHERE diagnosis_id = ?",
                (diagnosis_id,)).fetchone()
        return self._row(row) if row is not None else None

    def query(self, *, limit: int = 20, offset: int = 0,
              status: str | None = None, source_mode: str | None = None,
              model: str | None = None, text: str | None = None) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))
        where: list[str] = []
        args: list[Any] = []
        if status:
            where.append("diagnosis_status = ?")
            args.append(status)
        if source_mode:
            where.append("source_mode = ?")
            args.append(source_mode)
        if model:
            where.append("model_variant = ?")
            args.append(model)
        if text:
            where.append("(user_question LIKE ? OR fault_category LIKE ?)")
            pattern = f"%{text}%"
            args.extend([pattern, pattern])
        clause = " WHERE " + " AND ".join(where) if where else ""
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM diagnoses{clause}", tuple(args)).fetchone()[0]
            rows = connection.execute(
                f"SELECT * FROM diagnoses{clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                tuple(args + [limit, offset])).fetchall()
        return {
            "items": [self._row(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
            "schema_version": SCHEMA_VERSION,
        }

    def prune(self, *, retention_days: int, now: datetime | None = None) -> int:
        if retention_days <= 0:
            return 0
        cutoff = ((now or datetime.now(timezone.utc))
                  - timedelta(days=retention_days)).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM diagnoses WHERE created_at < ?", (cutoff,))
        return cursor.rowcount

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0])


def default_history_path() -> Path:
    configured = os.environ.get("PRINTDIAG_HISTORY_DB")
    if configured:
        return Path(configured)
    return Path("runs") / "p10_8c_history" / "diagnoses.sqlite3"
