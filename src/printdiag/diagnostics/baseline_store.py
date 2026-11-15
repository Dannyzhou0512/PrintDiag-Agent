"""Write each baseline to a separate directory with a manifest written last."""

import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from printdiag.diagnostics.device_discovery import JsonObject
from printdiag.diagnostics.state_collector import BaselineBatch


class BaselineStorageError(Exception):
    """The batch could not be written in full."""


def _json_text(data: JsonObject) -> str:
    return json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n"


def _write_json(path: Path, data: JsonObject) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(_json_text(data))


def save_baseline(
    batch: BaselineBatch,
    output_dir: Path,
    *,
    api_base_url: str,
    http_timeout_s: float,
) -> Path:
    manifest = batch.manifest
    identity = {key: manifest[key] for key in ("schema_version", "batch_id")}
    batch_dir = Path(output_dir) / identity["batch_id"]
    try:
        url = urlsplit(api_base_url)
        api_origin = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], "", "", ""))
        batch_dir.mkdir(parents=True, exist_ok=False)
        discovery = batch.discovery
        initial = discovery["initial_state"]
        first_success = next(
            (row for row in [initial, *batch.snapshots] if row["status"] == "ok"),
            None,
        )
        _write_json(batch_dir / "components.json", {
            **identity,
            "inventory": discovery["inventory"],
            "components": discovery["components"],
            "unprobed_names": discovery["unprobed_names"],
        })
        _write_json(batch_dir / "initial_state.json", {
            **identity,
            "snapshot": first_success,
            "discovery_failure": initial if initial["status"] != "ok" else None,
        })
        with (batch_dir / "snapshots.jsonl").open("x", encoding="utf-8", newline="\n") as output:
            for snapshot in batch.snapshots:
                output.write(_json_text({**identity, **snapshot}))
        _write_json(batch_dir / "gcode_queries.json", {**identity, "queries": batch.queries})
        manifest_tmp = batch_dir / "manifest.json.tmp"
        _write_json(manifest_tmp, {
            **manifest,
            "api_base_url": api_origin,
            "parameters": {**manifest["parameters"], "http_timeout_s": http_timeout_s},
        })
        manifest_tmp.rename(batch_dir / "manifest.json")
    except (OSError, TypeError, ValueError) as exc:
        raise BaselineStorageError(f"Cannot save baseline batch at {batch_dir}: {exc}") from exc
    return batch_dir
