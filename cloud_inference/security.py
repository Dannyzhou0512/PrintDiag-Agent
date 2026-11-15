"""Authentication helpers for the cloud inference service.

Tokens come from the environment only, comparison is constant-time, and no
token is ever logged. The service refuses to start without a strong token.
"""

import hmac
import os
import secrets

MIN_TOKEN_LENGTH = 16


def load_token_from_env(environ: dict[str, str] | None = None) -> str:
    env = environ if environ is not None else dict(os.environ)
    token = env.get("PRINTDIAG_CLOUD_TOKEN", "")
    if len(token) < MIN_TOKEN_LENGTH:
        raise ValueError(
            f"PRINTDIAG_CLOUD_TOKEN must be set with at least {MIN_TOKEN_LENGTH} characters")
    return token


def generate_token() -> str:
    """Helper for deployment docs; never written anywhere automatically."""
    return secrets.token_urlsafe(32)


def token_matches(candidate: str | None, secret: str) -> bool:
    if candidate is None:
        return False
    return hmac.compare_digest(candidate, secret)


def extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.casefold() != "bearer" or not value.strip():
        return None
    return value.strip()
