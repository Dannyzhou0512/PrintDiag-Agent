"""Load project settings from the environment and a local .env file."""

import os
from dataclasses import dataclass
from math import isfinite

from dotenv import dotenv_values


@dataclass(frozen=True)
class Config:
    marlin_base_url: str
    marlin_http_timeout: float
    log_level: str


def load_config() -> Config:
    """Read .env from the working directory and give environment variables precedence."""
    file_values = {
        key: value
        for key, value in dotenv_values(".env").items()
        if value is not None
    }
    values = {**file_values, **os.environ}

    timeout_value = values.get("MARLIN_HTTP_TIMEOUT", "5")
    timeout_error = (
        "MARLIN_HTTP_TIMEOUT must be a finite number greater than 0 "
        f"(seconds); got {timeout_value!r}."
    )
    try:
        timeout = float(timeout_value)
    except ValueError:
        raise ValueError(timeout_error) from None
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError(timeout_error)

    return Config(
        marlin_base_url=values.get("MARLIN_BASE_URL", "http://127.0.0.1:8100"),
        marlin_http_timeout=timeout,
        log_level=values.get("LOG_LEVEL", "INFO"),
    )
