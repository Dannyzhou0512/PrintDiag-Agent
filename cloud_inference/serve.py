"""Start the cloud inference service on the GPU host (127.0.0.1 only).

Usage on the cloud JupyterLab terminal:

    export PRINTDIAG_CLOUD_TOKEN='<long-random-token shared with Windows>'
    /data/venvs/qwen35/bin/python -m cloud_inference.serve

The service binds 127.0.0.1:8008; reach it from Windows through the existing
SSH tunnel (local 18008 -> remote 8008). Authentication is mandatory.
"""

import os
import sys


def main() -> int:
    token = os.environ.get("PRINTDIAG_CLOUD_TOKEN", "")
    if len(token) < 16:
        print("PRINTDIAG_CLOUD_TOKEN must be set (>=16 chars)", file=sys.stderr)
        return 2
    host = os.environ.get("PRINTDIAG_CLOUD_HOST", "127.0.0.1")
    port = int(os.environ.get("PRINTDIAG_CLOUD_PORT", "8008"))
    if host != "127.0.0.1":
        print("refusing to bind a non-loopback host", file=sys.stderr)
        return 2
    import uvicorn  # noqa: PLC0415

    uvicorn.run("cloud_inference.service:app", host=host, port=port, log_level="info",
                timeout_keep_alive=30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
