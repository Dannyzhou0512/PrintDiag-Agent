"""Append-only local audit for simulator control; never stores tokens."""

from datetime import UTC, datetime
import json
from pathlib import Path
import threading


class ControlAudit:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "control_audit.jsonl"
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        payload = {"utc": datetime.now(UTC).isoformat(), **record}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 30) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        rows = []
        for line in lines[-max(1, min(limit, 100)):]:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return list(reversed(rows))
