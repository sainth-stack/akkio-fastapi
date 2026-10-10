"""
LLM agents for the frontend-only pipeline stages.

Each agent:
  - Receives a minimal, focused context from context_builder.py
  - Returns generated file content
  - Never sees the full PRD text

Stage order:
  S1a  types.ts          — TypeScript interfaces from entities
  S1b  mock/index.ts     — realistic mock data + useMockStore hook
  S2   theme + layout    — deterministic (no LLM)
  S3   shared components — domain-specific shared pieces (may be empty)
  S4   pages             — one LLM call per page (parallel, bounded concurrency)
  S5   App.tsx/router    — deterministic (no LLM)
  S6   final verify      — no LLM
"""
from __future__ import annotations

import asyncio
import logging
import textwrap
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("app_builder")

# ---------------------------------------------------------------------------
# Allowed imports a page agent may use (enforced in prompt)
# ---------------------------------------------------------------------------

ALLOWED_IMPORTS = [
    "@mui/material",
    "@mui/icons-material",
    "react",
    "react-router-dom",
    "../components/ui",
    "../../components/ui",
    "../types",
    "../../types",
    "../mock",
    "../../mock",
    "../theme/tokens",
    "../../theme/tokens",
]

# ---------------------------------------------------------------------------
# S1a — TypeScript types
# ---------------------------------------------------------------------------

_TYPES_SYSTEM = """\
You are a senior TypeScript engineer. Generate a single file: src/types/index.ts.

Rules (STRICT):
1. One named `export interface` per entity (PascalCase singular).
2. Every interface has `id: string`.
3. Use proper TS types: string, number, boolean, Date, string[], number[].
4. For status/stage fields, also export a union type: `export type {Entity}Status = 'active'|'inactive'|...`
5. Add `createdAt?: string` and `updatedAt?: string` to all entities.
6. No imports — pure type definitions only.
7. Export a `mockKpis` type alias: `export type KpiItem = { label: string; value: string|number; trend?: { value: number; label: string } }`.
8. End the file with a `export type {}` barrel comment listing all exported names.

RETURN ONLY the complete TypeScript file. No markdown, no explanation."""


async def generate_types(
    context: Dict[str, Any],
    llm: Any,
    on_event: Optional[Callable] = None,
) -> str:
    """Generate src/types/index.ts from entity definitions."""
    entities = context.get("entities") or []
    domain = context.get("domain", "generic")
    app_name = context.get("app_name", "Application")

    entities_text = ""
    for ent in entities:
        entities_text += f"\n  Entity: {ent['name']}\n"
        for f in ent.get("fields") or []:
            entities_text += f"    - {f['name']}: {f['type']}\n"
        hints = ent.get("value_hints") or []
        if hints:
            entities_text += f"    Hints: {'; '.join(hints[:3])}\n"

    user_prompt = (
        f"Domain: {domain}\nApp: {app_name}\n\n"
        f"Entities:{entities_text or chr(10)+'  (no entities — create generic Item interface)'}\n\n"
        "Generate src/types/index.ts."
    )

    if on_event:
        await on_event({"event": "stage_file", "file": "frontend/src/types/index.ts", "status": "generating"})

    content = await _call_llm(llm, _TYPES_SYSTEM, user_prompt, label="types")
    return content or _fallback_types(entities)


# ---------------------------------------------------------------------------
# S1b — Mock data + useMockStore
# ---------------------------------------------------------------------------

_MOCK_SYSTEM = """\
You are a data engineer. Generate a single file: src/mock/index.ts.

Rules (STRICT):
1. Import ONLY from '../types' (use the types provided).
2. For each entity, export `const mock{Entity}: {Entity}[] = [...]` with 15-30 realistic rows.
3. Data must be domain-realistic (real names, dates, amounts, statuses — not "Item 1").
4. Consistent IDs: use '1','2',... strings; cross-entity relations must reference valid IDs.
5. Export `mockKpis: KpiItem[]` (4 items) and `mockChartData` (12-month series).
6. Export `useMockStore` hook (copy the pattern below EXACTLY):

```typescript
import { useState, useEffect } from 'react';

export function useMockStore<T extends { id: string }>(
  key: string,
  initial: T[]
) {
  const [items, setItems] = useState<T[]>(() => {
    try {
      const stored = localStorage.getItem(`mock_${key}`);
      return stored ? (JSON.parse(stored) as T[]) : initial;
    } catch { return initial; }
  });
  useEffect(() => {
    localStorage.setItem(`mock_${key}`, JSON.stringify(items));
  }, [items, key]);

  const add = (item: Omit<T, 'id'>): void =>
    setItems(p => [...p, { ...item, id: String(Date.now()) } as T]);
  const update = (id: string, patch: Partial<T>): void =>
    setItems(p => p.map(x => (x.id === id ? { ...x, ...patch } : x)));
  const remove = (id: string): void =>
    setItems(p => p.filter(x => x.id !== id));

  return { items, add, update, remove };
}
```

7. RETURN ONLY the complete TypeScript file. No markdown, no explanation."""


async def generate_mock(
    context: Dict[str, Any],
    llm: Any,
    on_event: Optional[Callable] = None,
) -> str:
    """Generate src/mock/index.ts with realistic data."""
    domain = context.get("domain", "generic")
    app_name = context.get("app_name", "Application")
    mock_plan = context.get("mock_plan") or []
    entities = context.get("entities") or []
    types_content = context.get("types_content") or ""

    # Build mock hints
    hints_text = ""
    for mp in mock_plan:
        entity = mp.get("entity", "")
        count = mp.get("row_count", 20)
        hints = mp.get("value_hints") or []
        hints_text += f"\n  {entity}: {count} rows, hints: {'; '.join(hints[:5])}"

    user_prompt = (
        f"Domain: {domain}\nApp: {app_name}\n\n"
        f"Mock plan:{hints_text or ' (no hints — use domain-realistic data)'}\n\n"
        f"Types file (import from here):\n```typescript\n{types_content[:2000]}\n```\n\n"
        "Generate src/mock/index.ts with realistic mock data and useMockStore."
    )

    if on_event:
        await on_event({"event": "stage_file", "file": "frontend/src/mock/index.ts", "status": "generating"})

    content = await _call_llm(llm, _MOCK_SYSTEM, user_prompt, label="mock")
    return content or _fallback_mock(entities, domain)


# ---------------------------------------------------------------------------
# S4 — Individual page generation
# ---------------------------------------------------------------------------

_PAGE_SYSTEM = """\
You are a senior React/TypeScript engineer building a production-quality SaaS page.

Rules (STRICT):
1. Write the complete src/pages/{page_name}.tsx file.
2. ONLY import from:
   - 'react' (useState, useEffect, useMemo, useCallback)
   - 'react-router-dom' (useNavigate, Link)
   - '@mui/material' (Box, Typography, Grid, Stack, Chip, IconButton, Tooltip, etc.)
   - '@mui/icons-material' (specific icon names)
   - '../components/ui' (kit components — use ONLY the signatures provided)
   - '../types' (entity interfaces — use ONLY the types provided)
   - '../mock' (mock exports — use ONLY the names provided)
   DO NOT invent new package names, component names, or type names.
3. Handle all 3 states explicitly:
   - Loading: wrap content with a loading check using <LoadingState />
   - Empty: show <EmptyState /> when the list is empty
   - Error: wrap with error boundary or inline check using <ErrorState />
4. For list/table screens: use <DataTable> with typed columns.
5. For add/edit operations: use <Modal> wrapping a form with <FormField> components.
6. For delete operations: use <ConfirmDialog> with destructive=true.
7. Show success/error feedback with <Toast>.
8. For charts: use the chart kit components with the mock chart data.
9. Use useMockStore for mutations (add/edit/delete).
10. The file must be complete — no TODO comments, no placeholder sections.
11. Use MUI sx props for styling — never hardcode hex colors.

RETURN ONLY the complete TypeScript file. No markdown fences, no explanation."""


async def generate_page(
    context: Dict[str, Any],
    llm: Any,
    on_event: Optional[Callable] = None,
) -> str:
    """Generate a single page component."""
    page_name = context.get("page_name", "Page")
    nav_label = context.get("nav_label", page_name.replace("Page", ""))
    ux_spec = context.get("ux_spec") or {}
    kit_signatures = context.get("kit_signatures") or {}
    relevant_types = context.get("relevant_types") or ""
    mock_exports = context.get("mock_exports") or []
    token_summary = context.get("token_summary") or ""
    domain = context.get("domain", "generic")

    kit_text = "\n".join(f"  {k}: {v}" for k, v in kit_signatures.items())
    mock_text = "  " + ", ".join(mock_exports) if mock_exports else "  (none — generate static UI)"

    ux_text = (
        f"Layout: {ux_spec.get('layout', 'default')}\n"
        f"Sections: {ux_spec.get('sections', [nav_label + ' content'])}\n"
        f"States — empty: {ux_spec.get('states', {}).get('empty', 'No items')}"
        f" | error: {ux_spec.get('states', {}).get('error', 'Error occurred')}\n"
        f"Actions: {ux_spec.get('actions', [])}"
    )

    system = _PAGE_SYSTEM.replace("{page_name}", page_name)

    user_prompt = (
        f"Page: {page_name} ({nav_label})\n"
        f"Domain: {domain}\n\n"
        f"UX spec:\n{ux_text}\n\n"
        f"Design tokens:\n{token_summary}\n\n"
        f"Types available (import from '../types'):\n{relevant_types[:1500]}\n\n"
        f"Mock exports available (import from '../mock'):\n{mock_text}\n\n"
        f"Kit component signatures (import from '../components/ui'):\n{kit_text}\n\n"
        f"Generate the complete {page_name}.tsx file."
    )

    if on_event:
        await on_event({
            "event": "stage_file",
            "file": f"frontend/src/pages/{page_name}.tsx",
            "status": "generating",
        })

    content = await _call_llm(llm, system, user_prompt, label=f"page:{page_name}")
    return content or _fallback_page(page_name, nav_label)


async def generate_pages_parallel(
    page_contexts: List[Dict[str, Any]],
    llm: Any,
    on_event: Optional[Callable] = None,
    max_concurrency: int = 3,
) -> Dict[str, str]:
    """Generate all pages in parallel, bounded by max_concurrency."""
    sem = asyncio.Semaphore(max_concurrency)
    results: Dict[str, str] = {}

    async def _one_page(ctx: Dict[str, Any]) -> None:
        page_name = ctx.get("page_name", "Page")
        async with sem:
            content = await generate_page(ctx, llm, on_event)
            results[f"frontend/src/pages/{page_name}.tsx"] = content

    await asyncio.gather(*[_one_page(ctx) for ctx in page_contexts])
    return results


# ---------------------------------------------------------------------------
# S3 — Shared domain components (optional)
# ---------------------------------------------------------------------------

_SHARED_SYSTEM = """\
You are a React engineer. Generate ONLY shared domain-specific sub-components
that are needed by multiple pages (e.g. StatusBadge, DealCard, PipelineBar).
DO NOT re-implement anything already in the kit (AppShell, DataTable, etc.).

Rules:
1. Each component goes in src/components/shared/{ComponentName}.tsx
2. Import ONLY from '@mui/material', '@mui/icons-material', '../ui', '../../types', 'react'.
3. If no shared components are genuinely needed, return exactly: NONE
4. RETURN each file as:
   === FILE: src/components/shared/ComponentName.tsx ===
   <file content>
   === END FILE ===
   Repeat for each file. If NONE, write exactly: NONE"""


async def generate_shared(
    blueprint_json: Dict[str, Any],
    types_content: str,
    llm: Any,
    on_event: Optional[Callable] = None,
) -> Dict[str, str]:
    """Generate optional domain-specific shared sub-components."""
    pages = blueprint_json.get("pages") or []
    domain = blueprint_json.get("domain") or "generic"
    app_name = blueprint_json.get("app_name") or "Application"

    # Collect unique kit components used across pages
    kit_usage: Dict[str, int] = {}
    for p in pages:
        for comp in (p.get("uses_kit_components") or []):
            kit_usage[comp] = kit_usage.get(comp, 0) + 1

    user_prompt = (
        f"Domain: {domain}\nApp: {app_name}\n"
        f"Pages: {[p.get('file') or p.get('name') for p in pages]}\n"
        f"Kit usage (component → page count): {dict(sorted(kit_usage.items(), key=lambda x: -x[1])[:8])}\n\n"
        "Are any domain-specific shared components needed? "
        "If yes, write them. If no, return NONE."
    )

    if on_event:
        await on_event({"event": "stage_file", "file": "frontend/src/components/shared/", "status": "generating"})

    content = await _call_llm(llm, _SHARED_SYSTEM, user_prompt, label="shared")

    if not content or content.strip().upper() == "NONE":
        return {}

    return _parse_file_blocks(content, prefix="frontend/src/components/")


# ---------------------------------------------------------------------------
# Fixer agent
# ---------------------------------------------------------------------------

_FIXER_SYSTEM = """\
You are a TypeScript debugger. Fix the provided file so it compiles cleanly.

Rules:
1. Return ONLY the COMPLETE, CORRECTED TypeScript file.
2. Fix ALL listed errors — do not introduce new ones.
3. Do not truncate — the file must be complete (all exports preserved).
4. Do not add any markdown fences or explanation.
5. Only change what is needed to fix the errors.
6. If an imported name doesn't exist, use an alternative from the available exports."""


async def fix_file(
    error_info: Dict[str, Any],
    types_content: str,
    mock_exports: List[str],
    llm: Any,
) -> Optional[str]:
    """
    Fix a broken file. Returns corrected content or None if LLM fails.
    Context is minimal: just the error + offending file + available types/exports.
    """
    error = error_info.get("error", "")
    offending_file = error_info.get("offending_file", "")
    file_content = error_info.get("file_content", "")
    all_errors = error_info.get("all_errors") or [error]

    errors_text = "\n".join(f"  {e}" for e in all_errors[:5])
    exports_text = "  " + ", ".join(mock_exports[:30]) if mock_exports else "  (none)"

    user_prompt = (
        f"File to fix: {offending_file}\n\n"
        f"Errors:\n{errors_text}\n\n"
        f"Available types (import from '../types'):\n"
        f"```typescript\n{types_content[:1000]}\n```\n\n"
        f"Available mock exports:\n{exports_text}\n\n"
        f"Broken file content:\n```typescript\n{file_content}\n```\n\n"
        "Return the complete corrected file."
    )

    return await _call_llm(llm, _FIXER_SYSTEM, user_prompt, label=f"fix:{offending_file}")


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

async def _call_llm(llm: Any, system: str, user: str, label: str = "") -> Optional[str]:
    """Invoke LLM and return string content. Non-streaming."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        # Strip markdown fences if present
        content = _strip_fences(content)
        logger.debug("[fo_stage/%s] LLM returned %d chars", label, len(content))
        return content
    except Exception as exc:
        logger.warning("[fo_stage/%s] LLM call failed: %s", label, exc)
        return None


def _strip_fences(text: str) -> str:
    """Remove ```typescript ... ``` or ``` ... ``` wrappers."""
    import re
    text = text.strip()
    # Remove opening fence with optional language tag
    text = re.sub(r"^```(?:typescript|tsx|ts|javascript|jsx|js)?\s*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# File block parser (for S3 multi-file response)
# ---------------------------------------------------------------------------

def _parse_file_blocks(content: str, prefix: str = "") -> Dict[str, str]:
    """
    Parse a response containing multiple file blocks:
    === FILE: path/to/file.tsx ===
    content
    === END FILE ===
    """
    import re
    files: Dict[str, str] = {}
    pattern = re.compile(
        r"===\s*FILE:\s*(.+?)\s*===\s*\n([\s\S]*?)\n===\s*END FILE\s*===",
        re.MULTILINE,
    )
    for m in pattern.finditer(content):
        path = m.group(1).strip()
        file_content = m.group(2).strip()
        if prefix and not path.startswith(prefix):
            # Normalize path to start with 'frontend/src/components/shared/'
            fname = os.path.basename(path)
            path = f"frontend/src/components/shared/{fname}"
        files[path] = file_content
    return files


import os  # needed by _parse_file_blocks


# ---------------------------------------------------------------------------
# Fallback generators (no LLM needed)
# ---------------------------------------------------------------------------

def _fallback_types(entities: List[Dict[str, Any]]) -> str:
    """Generate minimal types.ts without LLM."""
    lines = [
        "// Auto-generated types",
        "export type KpiItem = { label: string; value: string | number; trend?: { value: number; label: string } };",
        "",
    ]
    for ent in entities:
        name = ent.get("name", "Item")
        lines += [f"export interface {name} {{", "  id: string;"]
        for f in (ent.get("fields") or []):
            fn, ft = f.get("name", ""), f.get("type", "string")
            ts_type = {"Date": "string", "enum": "string"}.get(ft, ft)
            lines.append(f"  {fn}?: {ts_type};")
        lines += ["  createdAt?: string;", "  updatedAt?: string;", "}", ""]
    return "\n".join(lines)


def _fallback_mock(entities: List[str], domain: str) -> str:
    """Generate minimal mock/index.ts without LLM."""
    entity_lines: List[str] = []
    for ent in (entities or []):
        entity_lines += [
            f"export const mock{ent}s = [",
            f"  {{ id: '1', name: '{ent} Alpha', status: 'active', createdAt: '2026-01-10' }},",
            f"  {{ id: '2', name: '{ent} Beta', status: 'inactive', createdAt: '2026-02-15' }},",
            f"  {{ id: '3', name: '{ent} Gamma', status: 'active', createdAt: '2026-03-20' }},",
            "] as any[];",
            "",
        ]
    entity_block = "\n".join(entity_lines)

    return f"""\
import {{ useState, useEffect, useCallback }} from 'react';

// ---------------------------------------------------------------------------
// useMockStore — generic CRUD hook with localStorage persistence
// ---------------------------------------------------------------------------

export interface MockStoreResult<T> {{
  data: T[];
  loading: boolean;
  error: string | null;
  add: (item: T) => void;
  update: (id: string, patch: Partial<T>) => void;
  remove: (id: string) => void;
}}

export function useMockStore<T extends {{ id: string }}>(
  key: string = 'mock_store',
  seed: T[] = []
): MockStoreResult<T> {{
  const storageKey = `akkio_mock_${{key}}`;
  const [data, setData] = useState<T[]>(() => {{
    try {{
      const raw = localStorage.getItem(storageKey);
      if (raw) return JSON.parse(raw) as T[];
    }} catch {{}}
    return seed;
  }});
  const [loading] = useState(false);
  const [error] = useState<string | null>(null);

  useEffect(() => {{
    try {{ localStorage.setItem(storageKey, JSON.stringify(data)); }} catch {{}}
  }}, [data, storageKey]);

  const add = useCallback((item: T) => setData(p => [...p, item]), []);
  const update = useCallback((id: string, patch: Partial<T>) => {{
    setData(p => p.map(x => x.id === id ? {{ ...x, ...patch }} : x));
  }}, []);
  const remove = useCallback((id: string) => setData(p => p.filter(x => x.id !== id)), []);

  return {{ data, loading, error, add, update, remove }};
}}

export const mockKpis = [
  {{ label: 'Total', value: '1,234', trend: {{ value: 12, label: 'vs last month' }} }},
  {{ label: 'Active', value: '987', trend: {{ value: 5, label: 'vs last month' }} }},
  {{ label: 'Pending', value: '247', trend: {{ value: -3, label: 'vs last month' }} }},
  {{ label: 'Done', value: '456', trend: {{ value: 8, label: 'vs last month' }} }},
];

export const mockChartData = Array.from({{ length: 12 }}, (_, i) => ({{
  name: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][i],
  value: Math.floor(Math.random() * 400 + 100),
  previous: Math.floor(Math.random() * 400 + 80),
}}));

{entity_block}
"""


def _fallback_page(page_name: str, nav_label: str) -> str:
    """Generate a minimal placeholder page without LLM."""
    return textwrap.dedent(f"""\
        import {{ Box }} from '@mui/material';
        import {{ PageHeader, EmptyState }} from '../components/ui';

        export default function {page_name}() {{
          return (
            <Box>
              <PageHeader title="{nav_label}" />
              <EmptyState title="Coming Soon" description="This page is under construction." />
            </Box>
          );
        }}
        """)
