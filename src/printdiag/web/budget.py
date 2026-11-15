"""Persistent GPU budget ledger and live-request audit.

The ledger survives web-process restarts; only an explicit user action
(``scripts/start_printdiag_live.ps1 -ResetBudget``) clears it. It never stores
the token, prompt or model weights.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import threading


class BudgetError(RuntimeError):
    pass


class BudgetStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def configure(self, budget: int) -> dict:
        with self._lock:
            data = self._read()
            used = max(0, int(data.get("used", 0)))
            data.update({"budget": max(0, int(budget)), "used": used,
                         "updated_at": datetime.now(UTC).isoformat()})
            self._write(data)
            return dict(data)

    def status(self) -> dict:
        with self._lock:
            data = self._read()
            budget = max(0, int(data.get("budget", 0)))
            used = max(0, int(data.get("used", 0)))
            return {"budget": budget, "used": used,
                    "remaining": max(0, budget - used),
                    "path": str(self.path),
                    "updated_at": data.get("updated_at")}

    def reserve(self) -> dict:
        with self._lock:
            data = self._read()
            budget = max(0, int(data.get("budget", 0)))
            used = max(0, int(data.get("used", 0)))
            if used >= budget:
                raise BudgetError("LIVE_BUDGET_EXHAUSTED")
            used += 1
            data.update({"budget": budget, "used": used,
                         "updated_at": datetime.now(UTC).isoformat()})
            self._write(data)
            return {"budget": budget, "used": used,
                    "remaining": max(0, budget - used), "path": str(self.path)}

    def reset(self) -> None:
        with self._lock:
            data = self._read()
            self._write({"budget": max(0, int(data.get("budget", 0))), "used": 0,
                         "updated_at": datetime.now(UTC).isoformat()})


class AuditLogger:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "live_requests.jsonl"
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        payload = {"utc": datetime.now(UTC).isoformat(), **record}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
