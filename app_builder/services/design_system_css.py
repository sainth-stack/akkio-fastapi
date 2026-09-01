"""Design token extraction and SaaS CSS injection for generated apps."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app_builder.services.app_generators import build_saas_app_css


def extract_primary_hex(uiux: str, fallback: str = "#4f46e5") -> str:
    if not uiux:
        return fallback
    match = re.search(r"Primary:\s*(#[0-9A-Fa-f]{3,8})", uiux, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(#[0-9A-Fa-f]{6})", uiux)
    return match.group(1) if match else fallback


def normalize_token_payload(design_tokens: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not design_tokens or not isinstance(design_tokens, dict):
        return {}
    inner = design_tokens.get("design_tokens")
    if isinstance(inner, dict):
        return inner
    if design_tokens.get("colors"):
        return design_tokens
    return {}


def build_css_from_payload(
    design_tokens: Optional[Dict[str, Any]] = None,
    uiux: str = "",
) -> str:
    """Always produce a complete SaaS stylesheet from tokens and/or UI/UX palette."""
    tokens = normalize_token_payload(design_tokens)
    primary = (tokens.get("colors") or {}).get("primary") if tokens.get("colors") else None
    primary = str(primary or extract_primary_hex(uiux))
    if tokens:
        return build_saas_app_css(tokens, primary=primary)
    return build_saas_app_css(
        {
            "colors": {
                "primary": primary,
                "background": "#f8fafc",
                "surface": "#ffffff",
                "text": "#0f172a",
                "muted": "#64748b",
                "border": "#e2e8f0",
                "danger": "#ef4444",
            }
        },
        primary=primary,
    )
