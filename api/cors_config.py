"""Central CORS settings for the Akkio API."""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("akkio")

# localhost / 127.0.0.1 / LAN or public IPs with any port (http or https)
_DEV_ORIGIN_REGEX = (
    r"https?://("
    r"localhost|127\.0\.0\.1|\[::1\]"
    r"|(\d{1,3}\.){3}\d{1,3}"
    r")(:\d+)?"
)

_DEFAULT_DEV_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3002",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3002",
]


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def build_cors_middleware_kwargs() -> dict:
    """
    Build kwargs for CORSMiddleware.

    Priority:
    1. CORS_ALLOW_ALL=true  -> allow any origin (no credentials)
    2. CORS_ORIGINS + optional CORS_ORIGIN_REGEX from env
    3. In non-production, sensible localhost/IP defaults
    """
    env = os.getenv("ENV", "development").lower()
    allow_all = _truthy(os.getenv("CORS_ALLOW_ALL"))

    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    origin_regex = (os.getenv("CORS_ORIGIN_REGEX") or "").strip()

    if allow_all:
        logger.info("CORS: allow all origins (CORS_ALLOW_ALL=true, credentials disabled)")
        return {
            "allow_origins": ["*"],
            "allow_credentials": False,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
            "expose_headers": ["*"],
            "max_age": 600,
        }

    if env != "production":
        if not origins:
            origins = list(_DEFAULT_DEV_ORIGINS)
        if not origin_regex:
            origin_regex = _DEV_ORIGIN_REGEX
    elif not origins and not origin_regex:
        raise RuntimeError(
            "Set CORS_ORIGINS and/or CORS_ORIGIN_REGEX when ENV=production "
            "(or set CORS_ALLOW_ALL=true if appropriate for your deployment)."
        )

    kwargs: dict = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "expose_headers": ["*"],
        "allow_credentials": True,
        "max_age": 600,
    }
    if origins:
        kwargs["allow_origins"] = origins
    if origin_regex:
        re.compile(origin_regex)  # validate early
        kwargs["allow_origin_regex"] = origin_regex

    logger.info(
        "CORS: origins=%s regex=%s credentials=true",
        origins or "(none)",
        origin_regex or "(none)",
    )
    return kwargs
