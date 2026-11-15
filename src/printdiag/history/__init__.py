"""Local SQLite diagnosis history (P10-8C).

The module stores only validated report payloads and explicitly selected
metadata. It never stores API tokens, HTTP headers, environment dumps, model
raw output, or control audit files.
"""

from printdiag.history.store import HISTORY_SCHEMA_VERSION, HistoryRecordError, HistoryStore
from printdiag.history.service import (
    HistoryError,
    HistoryService,
    SOURCE_MODES,
    render_markdown,
)

__all__ = [
    "HISTORY_SCHEMA_VERSION",
    "HistoryError",
    "HistoryRecordError",
    "HistoryService",
    "HistoryStore",
    "SOURCE_MODES",
    "render_markdown",
]
