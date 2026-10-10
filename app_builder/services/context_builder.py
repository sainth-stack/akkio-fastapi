"""
Context builder for the frontend-only pipeline stage agents.

Each agent receives a *minimal*, focused context:
  - Never the full PRD text
  - Only the types/exports it actually imports
  - Only the kit component signatures it uses
  - Only the mock export names (not their full implementation)

This file is the single source of truth for what each agent sees.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Kit component signatures (compact — fit in one LLM context slot)
# ---------------------------------------------------------------------------

KIT_SIGNATURES: Dict[str, str] = {
    "AppShell": (
        "<AppShell navItems={NavItem[]} appName={string}>\n"
        "  {/* outlet */}\n"
        "</AppShell>\n"
        "// NavItem: { label: string; path: string; icon: string }"
    ),
    "PageHeader": (
        "<PageHeader title={string} subtitle?={string} "
        "actions?={ReactNode} />"
    ),
    "Button": (
        "<Button onClick={()=>void} loading?={boolean} "
        "variant?='contained'|'outlined'|'text' color?='primary'|'error'>"
        "label</Button>"
    ),
    "Card": (
        "<Card title?={string} subheader?={string} actions?={ReactNode}>"
        "{children}</Card>"
    ),
    "StatCard": (
        "<StatCard label={string} value={string|number} "
        "trend?={{ value: number; label: string }} accent?={string} />"
    ),
    "DataTable": (
        "<DataTable<T> data={T[]} columns={Column<T>[]} "
        "searchKeys?={string[]} loading?={boolean} />\n"
        "// Column<T>: { key: keyof T; label: string; render?: (row:T)=>ReactNode }"
    ),
    "FormField": (
        "<FormField label={string} value={string} "
        "onChange={(v:string)=>void} error?={string} "
        "type?={string} multiline?={boolean} rows?={number} />"
    ),
    "Modal": (
        "<Modal open={boolean} onClose={()=>void} title={string} "
        "actions?={ReactNode}>{children}</Modal>"
    ),
    "ConfirmDialog": (
        "<ConfirmDialog open={boolean} onClose={()=>void} "
        "onConfirm={()=>void} title={string} description?={string} "
        "destructive?={boolean} loading?={boolean} />"
    ),
    "Toast": (
        "<Toast open={boolean} onClose={()=>void} message={string} "
        "severity?='success'|'error'|'warning'|'info' />"
    ),
    "EmptyState": (
        "<EmptyState icon?={ReactNode} title={string} "
        "description?={string} action?={ReactNode} />"
    ),
    "LoadingState": "<LoadingState message?={string} />",
    "ErrorState": "<ErrorState message?={string} onRetry?={()=>void} />",
    "Tabs": (
        "<Tabs tabs={TabDef[]} />\n"
        "// TabDef: { label: string; content: ReactNode }"
    ),
    "LineChart": (
        "<LineChart data={Array<{name:string;[k:string]:any}>} "
        "lines={Array<{key:string;label:string;color?:string}>} "
        "xKey?={string} />"
    ),
    "BarChart": (
        "<BarChart data={Array<{name:string;[k:string]:any}>} "
        "bars={Array<{key:string;label:string;color?:string}>} "
        "xKey?={string} horizontal?={boolean} />"
    ),
    "PieChart": (
        "<PieChart data={Array<{name:string;value:number;color?:string}>} "
        "donut?={boolean} />"
    ),
    "StatusChip": "<StatusChip status={string} />",
}

# ---------------------------------------------------------------------------
# Context for S1 — types.ts generation
# ---------------------------------------------------------------------------

def build_types_context(blueprint_json: Dict[str, Any], prd_json: Optional[Dict]) -> Dict[str, Any]:
    """
    Context for the TypeScript types agent (Stage 1a).
    Contains ONLY entity definitions from the blueprint — NOT the PRD prose.
    """
    entities = blueprint_json.get("entities") or []
    mock_plan = blueprint_json.get("mock_plan") or []
    domain = blueprint_json.get("domain") or "generic"
    # Merge mock hints into entity list for richer field types
    hints_by_entity = {m["entity"]: m.get("value_hints", []) for m in mock_plan if m.get("entity")}

    enriched_entities = []
    for ent in entities:
        name = ent.get("name", "")
        fields = ent.get("fields", [])
        hints = hints_by_entity.get(name, [])
        enriched_entities.append({
            "name": name,
            "fields": fields,
            "value_hints": hints,
        })

    return {
        "domain": domain,
        "app_name": blueprint_json.get("app_name") or "Application",
        "entities": enriched_entities,
    }


# ---------------------------------------------------------------------------
# Context for S1b — mock data generation
# ---------------------------------------------------------------------------

def build_mock_context(
    blueprint_json: Dict[str, Any],
    types_content: str,
) -> Dict[str, Any]:
    """
    Context for the mock data agent.
    Passes the types file content (to import from) + mock plan hints.
    """
    mock_plan = blueprint_json.get("mock_plan") or []
    entities = [e.get("name", "") for e in (blueprint_json.get("entities") or [])]

    return {
        "domain": blueprint_json.get("domain") or "generic",
        "app_name": blueprint_json.get("app_name") or "Application",
        "entities": entities,
        "mock_plan": mock_plan,
        "types_content": types_content,  # full types.ts for interface names
    }


# ---------------------------------------------------------------------------
# Context for S4 — individual page generation
# ---------------------------------------------------------------------------

def build_page_context(
    page_file: str,               # e.g. "DashboardPage"
    blueprint_json: Dict[str, Any],
    uiux_json: Optional[Dict],    # UXPlan dict
    design_tokens_json: Optional[Dict],
    types_content: str,           # src/types/index.ts content
    mock_exports: List[str],      # exported names from mock/index.ts
) -> Dict[str, Any]:
    """
    Build a minimal, focused context for a single page agent.

    CONTRACT: The full PRD text is NEVER included here.
    Only the screen's UX spec + relevant types + mock exports + kit sigs.
    """
    # 1. Find this page in blueprint
    pages = blueprint_json.get("pages") or []
    page_spec = next(
        (p for p in pages if p.get("file") == page_file or p.get("name") == page_file),
        None,
    )
    if page_spec is None:
        # Fallback: use page_file as the name
        page_spec = {"file": page_file, "uses_entities": [], "uses_kit_components": []}

    # 2. Find the UX slice for this page's screen
    screen_id = None
    routes = blueprint_json.get("routes") or []
    for r in routes:
        if r.get("page") == page_file:
            screen_id = r.get("screen_id")
            break

    ux_slice: Optional[Dict] = None
    if uiux_json and screen_id:
        for sx in uiux_json.get("screens") or []:
            if sx.get("screen_id") == screen_id:
                ux_slice = sx
                break

    # 3. Kit signatures — only the ones this page uses
    used_kit = page_spec.get("uses_kit_components") or []
    if not used_kit and ux_slice:
        used_kit = ux_slice.get("components_from_kit") or []
    kit_sigs = {k: KIT_SIGNATURES[k] for k in used_kit if k in KIT_SIGNATURES}
    # Always include core structural components
    for always in ("AppShell", "PageHeader", "EmptyState", "LoadingState", "ErrorState"):
        if always not in kit_sigs:
            kit_sigs[always] = KIT_SIGNATURES[always]

    # 4. Relevant types — only what this page uses
    used_entities = page_spec.get("uses_entities") or []
    relevant_types = _extract_relevant_types(types_content, used_entities)

    # 5. Mock exports this page can use
    relevant_mocks = _filter_mock_exports(mock_exports, used_entities)

    # 6. Design token summary (brief, not full JSON)
    token_summary = _summarize_tokens(design_tokens_json)

    return {
        "page_name": page_file,
        "nav_label": page_spec.get("nav_label") or page_file.replace("Page", ""),
        "screen_id": screen_id or "",
        "ux_spec": ux_slice or {
            "layout": "default",
            "sections": [f"{page_file.replace('Page','')} content"],
            "components_from_kit": list(used_kit),
            "states": {
                "empty": "No data yet",
                "loading": "Loading...",
                "error": "Something went wrong",
            },
            "actions": [],
        },
        "used_entities": used_entities,
        "relevant_types": relevant_types,
        "kit_signatures": kit_sigs,
        "mock_exports": relevant_mocks,
        "token_summary": token_summary,
        # Metadata (never the full PRD text)
        "domain": blueprint_json.get("domain") or "generic",
        "app_name": blueprint_json.get("app_name") or "Application",
    }


# ---------------------------------------------------------------------------
# Context for the fixer agent (S4 failure recovery)
# ---------------------------------------------------------------------------

def build_fixer_context(
    error_message: str,
    offending_file: str,
    offending_content: str,
    types_content: str,
    mock_exports: List[str],
) -> Dict[str, Any]:
    """
    Minimal context for the fixer agent.
    Only the error + offending file + relevant types.
    """
    return {
        "error": error_message,
        "file": offending_file,
        "content": offending_content,
        "types_content": types_content,
        "mock_exports": mock_exports,
    }


# ---------------------------------------------------------------------------
# Context size guard (used in tests)
# ---------------------------------------------------------------------------

def assert_no_prd_in_context(context: Dict[str, Any], prd_text: str) -> None:
    """Raises AssertionError if the full PRD text leaks into any context value."""
    if not prd_text or len(prd_text) < 50:
        return
    # Use a distinctive 30-char substring to detect leakage
    fingerprint = prd_text[:30].strip()
    context_str = json.dumps(context)
    if fingerprint in context_str:
        raise AssertionError(
            f"PRD text leaked into agent context! fingerprint={fingerprint!r}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_relevant_types(types_content: str, entity_names: List[str]) -> str:
    """
    Extract only the interface/type blocks for the given entity names.
    Falls back to the full types_content if extraction fails.
    """
    if not entity_names or not types_content:
        return types_content or ""

    import re
    lines = types_content.splitlines()
    result_blocks: List[str] = []

    # Also include useMockStore since most pages use it
    result_blocks.append("// import { useMockStore } from '../mock';")

    for entity in entity_names:
        # Match: export interface Foo { ... } (possibly multi-line)
        pattern = re.compile(
            r"(export\s+(?:interface|type)\s+" + re.escape(entity) + r"[\s\S]*?\})",
            re.MULTILINE,
        )
        for match in pattern.finditer(types_content):
            result_blocks.append(match.group(0))

    return "\n\n".join(result_blocks) if result_blocks else types_content


def _filter_mock_exports(mock_exports: List[str], entity_names: List[str]) -> List[str]:
    """Return mock exports relevant to the given entities + always-useful ones."""
    always = {"mockKpis", "mockChartData", "useMockStore"}
    relevant = set(always)
    for name in entity_names:
        # mock{Entity}, use{Entity}Store variations
        lower = name.lower()
        for export in mock_exports:
            if lower in export.lower():
                relevant.add(export)
    return sorted(relevant & set(mock_exports)) + sorted(
        always - set(mock_exports)
    )


def _summarize_tokens(design_tokens_json: Optional[Dict]) -> str:
    """Return a brief token summary for use in page prompts."""
    if not design_tokens_json:
        return "Use MUI sx props with theme.palette colors (primary.main, background.default, etc.)"

    light = design_tokens_json.get("light") or {}
    font = design_tokens_json.get("font_family") or "Inter"
    radius = design_tokens_json.get("border_radius") or "8px"
    primary = light.get("primary") or "#1976d2"
    bg = light.get("background") or "#f5f7fb"
    surface = light.get("surface") or "#ffffff"

    return (
        f"Font: {font} | BorderRadius: {radius} | "
        f"Primary: {primary} | Background: {bg} | Surface: {surface}\n"
        "Use MUI sx props ONLY — do not hardcode hex colors in JSX. "
        "Reference via theme.palette or pass as sx={{ bgcolor: 'background.paper' }}."
    )
