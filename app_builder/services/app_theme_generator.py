"""Deterministic per-app visual identity — colors, charts, shell variant (not one static blue template)."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from app_builder.schemas.app_definition import AppDefinition, AppTheme

_DEFAULT_PRIMARY = "#1976d2"

# Distinct SaaS palettes (primary, dark, light, secondary, bg, surface, chart palette)
_THEME_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "ocean",
        "primary": "#0ea5e9",
        "primaryDark": "#0369a1",
        "primaryLight": "#7dd3fc",
        "secondary": "#6366f1",
        "background": "#f0f9ff",
        "surface": "#ffffff",
        "chartPalette": ["#0ea5e9", "#6366f1", "#14b8a6", "#f59e0b", "#ec4899", "#64748b"],
        "shellVariant": "brand",
    },
    {
        "id": "forest",
        "primary": "#059669",
        "primaryDark": "#047857",
        "primaryLight": "#6ee7b7",
        "secondary": "#0d9488",
        "background": "#f0fdf4",
        "surface": "#ffffff",
        "chartPalette": ["#059669", "#0d9488", "#84cc16", "#ca8a04", "#2563eb", "#64748b"],
        "shellVariant": "light",
    },
    {
        "id": "violet",
        "primary": "#7c3aed",
        "primaryDark": "#5b21b6",
        "primaryLight": "#c4b5fd",
        "secondary": "#db2777",
        "background": "#f5f3ff",
        "surface": "#ffffff",
        "chartPalette": ["#7c3aed", "#db2777", "#2563eb", "#f59e0b", "#14b8a6", "#64748b"],
        "shellVariant": "brand",
    },
    {
        "id": "slate",
        "primary": "#334155",
        "primaryDark": "#1e293b",
        "primaryLight": "#94a3b8",
        "secondary": "#0ea5e9",
        "background": "#f8fafc",
        "surface": "#ffffff",
        "chartPalette": ["#334155", "#0ea5e9", "#22c55e", "#eab308", "#ef4444", "#a855f7"],
        "shellVariant": "brand",
    },
    {
        "id": "ember",
        "primary": "#ea580c",
        "primaryDark": "#c2410c",
        "primaryLight": "#fdba74",
        "secondary": "#dc2626",
        "background": "#fff7ed",
        "surface": "#ffffff",
        "chartPalette": ["#ea580c", "#dc2626", "#ca8a04", "#2563eb", "#059669", "#64748b"],
        "shellVariant": "light",
    },
    {
        "id": "rose",
        "primary": "#e11d48",
        "primaryDark": "#be123c",
        "primaryLight": "#fda4af",
        "secondary": "#7c3aed",
        "background": "#fff1f2",
        "surface": "#ffffff",
        "chartPalette": ["#e11d48", "#7c3aed", "#2563eb", "#059669", "#f59e0b", "#64748b"],
        "shellVariant": "light",
    },
    {
        "id": "midnight",
        "primary": "#1d4ed8",
        "primaryDark": "#1e3a8a",
        "primaryLight": "#60a5fa",
        "secondary": "#06b6d4",
        "background": "#eff6ff",
        "surface": "#ffffff",
        "chartPalette": ["#1d4ed8", "#06b6d4", "#22c55e", "#f97316", "#a855f7", "#64748b"],
        "shellVariant": "brand",
    },
    {
        "id": "teal",
        "primary": "#0f766e",
        "primaryDark": "#115e59",
        "primaryLight": "#5eead4",
        "secondary": "#0284c7",
        "background": "#f0fdfa",
        "surface": "#ffffff",
        "chartPalette": ["#0f766e", "#0284c7", "#84cc16", "#f59e0b", "#e11d48", "#64748b"],
        "shellVariant": "light",
    },
]

# Soft pools — hash picks one preset per app (avoids every manufacturing app = same midnight blue).
_PRESET_POOLS: List[Tuple[str, List[str]]] = [
    (r"health|clinical|hospital|patient", ["teal", "forest", "ocean"]),
    (r"finance|bank|ledger|payment|fintech", ["midnight", "slate", "violet"]),
    (r"logistics|supply|warehouse|inventory|fleet", ["slate", "ocean", "midnight"]),
    (r"security|cyber|threat", ["violet", "slate", "midnight"]),
    (r"energy|utility|grid|solar", ["forest", "teal", "ocean"]),
    (r"retail|commerce|shop", ["rose", "ember", "ocean"]),
    (
        r"manufactur|maintenance|factory|plant|industrial|quality|oee|navy|monitoring",
        ["midnight", "slate", "teal", "ocean", "forest", "violet"],
    ),
    (r"analytics|data|insight", ["ocean", "violet", "slate", "midnight"]),
]

_GENERIC_PRIMARIES = frozenset(
    {
        "#1976d2",
        "#1565c0",
        "#1d4ed8",
        "#1e3a8a",
        "#0b3d6f",
        "#082847",
        "#2563eb",
        "#4f46e5",
        "#0ea5e9",
    }
)


def _seed_index(*parts: str) -> int:
    raw = "|".join(p.strip() for p in parts if p).encode("utf-8")
    h = hashlib.sha256(raw or b"akkio").hexdigest()
    return int(h[:8], 16)


def _preset_by_id(pid: str) -> Optional[Dict[str, Any]]:
    for p in _THEME_PRESETS:
        if p["id"] == pid:
            return p
    return None


def _candidate_preset_ids(blob: str) -> List[str]:
    for pattern, pool in _PRESET_POOLS:
        if re.search(pattern, blob):
            return pool
    return [p["id"] for p in _THEME_PRESETS]


def _pick_preset(
    requirement: str,
    domain: str,
    app_name: str,
    *,
    variation_seed: str = "",
) -> Dict[str, Any]:
    blob = f"{requirement} {domain} {app_name}".lower()
    pool = _candidate_preset_ids(blob)
    idx = _seed_index(app_name, domain, requirement[:800], variation_seed) % len(pool)
    preset_id = pool[idx]
    found = _preset_by_id(preset_id)
    if found:
        return found
    return _THEME_PRESETS[_seed_index(variation_seed, app_name) % len(_THEME_PRESETS)]


def theme_is_default(theme: Optional[AppTheme]) -> bool:
    if theme is None:
        return True
    return (theme.primary or "").lower() == _DEFAULT_PRIMARY.lower()


def theme_looks_generic(theme: Optional[AppTheme]) -> bool:
    if theme is None:
        return True
    return (theme.primary or "").lower() in _GENERIC_PRIMARIES


def generate_app_theme(
    *,
    app_name: str = "App",
    domain: str = "generic",
    requirement: str = "",
    description: str = "",
    variation_seed: str = "",
) -> AppTheme:
    preset = _pick_preset(
        f"{requirement} {description}",
        domain,
        app_name,
        variation_seed=variation_seed,
    )
    fonts = ("Inter", "DM Sans", "Plus Jakarta Sans", "Source Sans 3")
    font_idx = _seed_index(app_name, "font", variation_seed) % len(fonts)
    return AppTheme(
        primary=preset["primary"],
        primaryDark=preset["primaryDark"],
        primaryLight=preset["primaryLight"],
        secondary=preset["secondary"],
        background=preset["background"],
        surface=preset["surface"],
        text="#0f172a",
        muted="#64748b",
        danger="#dc2626",
        success="#16a34a",
        warning="#d97706",
        info=preset["secondary"],
        fontFamily=fonts[font_idx],
        chartPalette=list(preset["chartPalette"]),
        shellVariant=preset.get("shellVariant", "light"),
        border="#e2e8f0",
    )


def merge_design_tokens_into_theme(
    theme: AppTheme,
    design_tokens: Optional[Dict[str, Any]],
    *,
    preserve_generated_colors: bool = True,
) -> AppTheme:
    if not design_tokens:
        return theme
    from app_builder.services.app_definition_compiler import _theme_from_tokens

    override = _theme_from_tokens(design_tokens)
    data = theme.model_dump()
    o = override.model_dump()
    token_primary = (o.get("primary") or "").lower()
    skip_colors = preserve_generated_colors and token_primary in _GENERIC_PRIMARIES
    color_keys = frozenset(
        {"primary", "primaryDark", "primaryLight", "secondary", "background", "surface", "chartPalette"}
    )
    for key, val in o.items():
        if not val:
            continue
        if skip_colors and key in color_keys:
            continue
        if key == "chartPalette" and isinstance(val, list) and val:
            data[key] = val
        elif val:
            data[key] = val
    return AppTheme(**data)


_GENERIC_CHART_HEX = frozenset(
    {
        "#2563eb",
        "#16a34a",
        "#ea580c",
        "#7c3aed",
        "#0891b2",
        "#dc2626",
        "#64748b",
        "#1976d2",
        "#2e7d32",
        "#ed6c02",
    }
)


def apply_theme_to_chart_blocks(
    defn: AppDefinition,
    *,
    replace_generic_colors: bool = False,
) -> AppDefinition:
    palette = list(defn.theme.chartPalette or [])
    if not palette:
        return defn
    for page in defn.pages:
        for block in page.blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") not in ("lineChart", "barChart"):
                continue
            series = block.get("series")
            if not isinstance(series, list):
                continue
            for i, s in enumerate(series):
                if not isinstance(s, dict):
                    continue
                existing = str(s.get("color") or "").strip().lower()
                if replace_generic_colors and existing in _GENERIC_CHART_HEX:
                    s["color"] = palette[i % len(palette)]
                elif not s.get("color"):
                    s["color"] = palette[i % len(palette)]
    return defn


def ensure_app_visual_identity(
    defn: AppDefinition,
    *,
    requirement: str = "",
    design_tokens: Optional[Dict[str, Any]] = None,
    variation_seed: str = "",
) -> AppDefinition:
    """Assign a varied theme per build; sync chart series colors."""
    if theme_is_default(defn.theme) or theme_looks_generic(defn.theme):
        defn.theme = generate_app_theme(
            app_name=defn.meta.appName,
            domain=defn.meta.domain or "generic",
            requirement=requirement,
            description=defn.meta.description or "",
            variation_seed=variation_seed,
        )
    defn.theme = merge_design_tokens_into_theme(
        defn.theme,
        design_tokens,
        preserve_generated_colors=True,
    )
    return apply_theme_to_chart_blocks(defn)


def ensure_theme_variation_seed(plan_json: Optional[Dict[str, Any]]) -> str:
    """Stable per codegen run; new seed on each hydrate when missing."""
    import uuid

    plan = plan_json if isinstance(plan_json, dict) else {}
    existing = str(plan.get("theme_variation_seed") or "").strip()
    if existing:
        return existing
    return uuid.uuid4().hex[:16]
