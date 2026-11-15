#!/usr/bin/env bash
# Start the PrintDiag cloud inference service on the GPU host.
# Default: foreground (Ctrl+C stops). Use --background for a PID file + log.
set -euo pipefail

APP_DIR="${PRINTDIAG_CLOUD_APP_DIR:-/data/projects/printdiag-cloud-inference}"
PYTHON="${PRINTDIAG_CLOUD_PYTHON:-/data/venvs/qwen35/bin/python}"
PORT="${PRINTDIAG_CLOUD_PORT:-8008}"
BASE="${PRINTDIAG_CLOUD_BASE:-/data/models/Qwen3.5-4B}"
ADAPTER="${PRINTDIAG_CLOUD_ADAPTER:-/data/outputs/qwen35_4b_lora_v03}"

die() { echo "ERROR: $*" >&2; exit 2; }

[ "${PRINTDIAG_CLOUD_TOKEN:-}" ] || die "PRINTDIAG_CLOUD_TOKEN is not set"
[ "${#PRINTDIAG_CLOUD_TOKEN}" -ge 16 ] || die "PRINTDIAG_CLOUD_TOKEN must be >= 16 chars"
[ -x "$PYTHON" ] || die "python not found: $PYTHON"
[ -f "$BASE/config.json" ] || die "base model missing: $BASE/config.json"
[ -f "$ADAPTER/adapter_config.json" ] || die "adapter missing: $ADAPTER/adapter_config.json"
[ -f "$ADAPTER/adapter_model.safetensors" ] || die "adapter weights missing: $ADAPTER"
[ -d "$APP_DIR/cloud_inference" ] || die "cloud_inference package missing in $APP_DIR"

"$PYTHON" - <<'PY' || die "missing web/runtime dependency (see requirements-cloud.txt)"
import importlib.util as u
missing = [m for m in ("fastapi", "uvicorn", "pydantic", "torch", "transformers", "peft")
           if u.find_spec(m) is None]
if missing:
    print("missing:", ", ".join(missing))
    raise SystemExit(1)
PY

cd "$APP_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ "${1:-}" = "--background" ]; then
  mkdir -p logs
  nohup "$PYTHON" -m uvicorn cloud_inference.service:app \
      --host 127.0.0.1 --port "$PORT" >> logs/server.log 2>&1 &
  echo $! > server.pid
  echo "started pid $(cat server.pid) on 127.0.0.1:${PORT}; log: $APP_DIR/logs/server.log"
  echo "first start loads Base + LoRA (about 1-3 min, ~17 GB BF16 on RTX 5090)"
else
  echo "starting foreground on 127.0.0.1:${PORT}; Ctrl+C to stop"
  exec "$PYTHON" -m uvicorn cloud_inference.service:app --host 127.0.0.1 --port "$PORT"
fi
