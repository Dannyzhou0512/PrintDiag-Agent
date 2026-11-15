"""Cloud service entrypoint (uvicorn target: ``cloud_inference.server:app``).

Deployment uses ``cloud_inference.service:app``; this module is the documented
stable alias and CLI entrypoint.
"""

from cloud_inference.service import (  # noqa: F401
    SERVICE_NAME, SERVICE_VERSION, app, create_cloud_app,
)

__all__ = ["SERVICE_NAME", "SERVICE_VERSION", "app", "create_cloud_app"]


if __name__ == "__main__":
    from cloud_inference.serve import main

    raise SystemExit(main())
