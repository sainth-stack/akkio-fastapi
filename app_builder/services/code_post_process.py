"""Shared post-processing for generated application files (codegen + LangGraph)."""

from __future__ import annotations

import ast
import logging
import re
from typing import Any, Dict, Optional, Tuple

from app_builder.agents import dynamic_code_generator as dcg
from app_builder.services.app_spec_service import (
    build_app_spec,
    get_codegen_allowlist,
    validate_against_app_spec,
)
from app_builder.services.scaffold_service import (
    get_base_scaffold_files,
    is_frozen_path,
    merge_llm_into_base,
)
from app_builder.services.app_generators import apply_deterministic_fallback, generate_llm_app_jsx
from app_builder.services.design_system_css import build_css_from_payload

logger = logging.getLogger("app_builder")


def _validate_backend_python_syntax(files: Dict[str, str]) -> None:
    """Catch syntax errors before backend verify / deploy (SaaS-quality gate)."""
    from api.app_creator.backend_verify_service import apply_deterministic_backend_fixes

    apply_deterministic_backend_fixes(files)
    for path in list(files.keys()):
        if not (path.startswith("backend/") and path.endswith(".py")):
            continue
        try:
            ast.parse(files[path])
        except SyntaxError as exc:
            logger.warning("[codegen] invalid Python in %s: %s — dropping file for re-merge", path, exc)
            files.pop(path, None)


def _is_vite_project(files: Dict[str, str]) -> bool:
    pkg = files.get("frontend/package.json", "")
    if "vite" in pkg.lower():
        return True
    return "frontend/vite.config.js" in files or "frontend/vite.config.ts" in files


def apply_design_tokens(
    files: Dict[str, str],
    design_tokens: Optional[Dict[str, Any]] = None,
    uiux: str = "",
) -> None:
    """Inject complete SaaS CSS from design tokens (always overwrites LLM app.css)."""
    css = build_css_from_payload(design_tokens, uiux=uiux)
    files["frontend/src/styles/app.css"] = css
    logger.info("[css] applied SaaS design system (%d lines)", len(css.splitlines()))


def _ensure_llm_app_jsx(files: Dict[str, str], app_spec: Dict[str, Any]) -> None:
    """Replace broken LLM codegen JSX with the deterministic travel/summary template."""
    if app_spec.get("app_kind") != "llm":
        return
    jsx = files.get("frontend/src/App.jsx") or files.get("frontend/src/App.js", "")
    if not jsx:
        return
    needs_fix = (
        ".json()" in jsx
        or (
            (app_spec.get("gen_type") or "general") in ("travel", "summarize", "general", "linkedin_post", "translate")
            and "formatMarkdown" not in jsx
        )
    )
    if not needs_fix:
        return
    title_match = re.search(r'<h1[^>]*className="app-title"[^>]*>([^<]+)</h1>', jsx)
    title = title_match.group(1).strip() if title_match else "AI App"
    files["frontend/src/App.jsx"] = generate_llm_app_jsx(app_spec, title=title)
    logger.info("[codegen] replaced broken LLM App.jsx with deterministic template (gen_type=%s)", app_spec.get("gen_type"))


def post_process_generated_files(
    files: Dict[str, str],
    architecture: Optional[Dict[str, Any]] = None,
    template_name: Optional[str] = None,
    uiux: str = "",
    requirement: str = "",
    prd: str = "",
    app_spec: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
    builder_kind: Optional[str] = None,
    project_name: str = "",
) -> Dict[str, str]:
    """Apply normalization/fixups. Merges LLM output into generic base shell."""
    from app_builder.services.fullstack_codegen import is_fullstack_generated_files, post_process_fullstack_files
    from app_builder.services.fullstack_stack import is_fullstack

    files_in = dict(files or {})
    if is_fullstack(builder_kind) or is_fullstack_generated_files(files_in):
        return post_process_fullstack_files(
            files_in,
            design_tokens=design_tokens,
            uiux=uiux,
            requirement=requirement,
            prd=prd,
            project_name=project_name,
        )

    spec = app_spec or build_app_spec(requirement, architecture, prd, uiux)
    allowlist = get_codegen_allowlist(spec)
    base = get_base_scaffold_files()
    files = dict(files or {})
    files = merge_llm_into_base(base, files, allowlist=allowlist, app_spec=spec)

    if not _is_vite_project(files):
        dcg._normalize_frontend_package_json_to_cra(files)

    dcg._ensure_node_compat(files)
    dcg._fix_frontend_map_safety(files)
    dcg._fix_backend_routes_import(files)
    if not _is_vite_project(files):
        dcg._ensure_vite_react_plugin_in_package_json(files)
    dcg._normalize_backend_requirements(files)
    dcg._ensure_sqlite_database_default(files)
    dcg._fix_backend_python_relative_imports(files)
    dcg._fix_sqlalchemy_uuid_imports(files)
    dcg._fix_backend_pydantic_and_common(files)
    _validate_backend_python_syntax(files)
    dcg._ensure_database_tables_created(files)
    dcg._ensure_cors_in_backend(files)
    dcg._fix_frontend_backend_url_undefined(files)
    dcg._validate_and_fix_backend_imports(files)
    apply_design_tokens(files, design_tokens, uiux=uiux)
    dcg._ensure_complete_styles_css(files, uiux=uiux)

    _ensure_llm_app_jsx(files, spec)
    dcg._ensure_jsx_css_alignment(files)

    for path in list(files.keys()):
        if is_frozen_path(path) and path in base:
            files[path] = base[path]

    if _is_vite_project(files):
        for junk in (
            "frontend/craco.config.js",
            "frontend/public/index.html",
            "frontend/src/index.js",
        ):
            files.pop(junk, None)

    return files


def should_use_template(requirement: str, prd: str, architecture: Optional[dict]) -> bool:
    return True


def validate_codegen_output(
    files: Dict[str, str],
    architecture: Optional[Dict[str, Any]] = None,
    requirement: str = "",
    prd: str = "",
    app_spec: Optional[Dict[str, Any]] = None,
) -> list[str]:
    spec = app_spec or build_app_spec(requirement, architecture, prd)
    return validate_against_app_spec(files, spec)


def ensure_valid_codegen_output(
    files: Dict[str, str],
    architecture: Optional[Dict[str, Any]] = None,
    uiux: str = "",
    requirement: str = "",
    prd: str = "",
    app_spec: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], list[str]]:
    """
    Validate generated files. Apply deterministic fallback for known app kinds when LLM fails.
    """
    spec = app_spec or build_app_spec(requirement, architecture, prd, uiux)
    errors = validate_against_app_spec(files, spec)
    if not errors:
        return files, []

    kind = spec.get("app_kind", "custom")

    # Complex multi-page apps: only fall back if the LLM generated nothing at all.
    # If App.jsx has real content (> 30 lines), trust the LLM — don't replace with generic CRUD.
    if kind == "complex":
        app_src = files.get("frontend/src/App.jsx") or files.get("frontend/src/App.js", "")
        has_real_frontend = len(app_src.splitlines()) > 30 and any(
            k in app_src for k in ("useState", "onClick", "return", "<div", "function")
        )
        if has_real_frontend:
            logger.info(
                "[codegen] complex app has real frontend (%d lines) — skipping deterministic fallback; errors: %s",
                len(app_src.splitlines()),
                "; ".join(errors[:3]),
            )
            return files, []

    supported = (
        "static_quiz", "static", "quiz", "crud", "content", "form", "llm", "dashboard", "custom", "complex",
    )
    if errors and (kind in supported or spec.get("frontend_only")):
        logger.warning(
            "[codegen] LLM validation failed (%s) — applying deterministic %s fallback",
            "; ".join(errors[:3]),
            kind,
        )
        fallback_files = apply_deterministic_fallback(dict(files), spec, uiux=uiux, prd=prd)
        fallback_files = post_process_generated_files(
            fallback_files,
            architecture,
            uiux=uiux,
            requirement=requirement,
            prd=prd,
            app_spec=spec,
            design_tokens=design_tokens,
        )
        fb_errors = validate_against_app_spec(fallback_files, spec)
        if not fb_errors:
            return fallback_files, []

    logger.warning("[codegen] validation failed: %s", "; ".join(errors[:5]))
    return files, errors


# ---------------------------------------------------------------------------
# Frontend-only pipeline page fixer (deterministic, no LLM needed)
# ---------------------------------------------------------------------------

_CHART_WRONG_IMPORT_RE = re.compile(
    r"import\s+\{\s*(LineChart|BarChart|PieChart)\s*\}\s+from\s+'[^']*charts/(?:LineChart|BarChart|PieChart)'",
)
_CHART_BARREL_IMPORT_RE = re.compile(
    r"(import\s*\{[^}]*)\}\s*from\s*'../components/ui'",
)


def fix_frontend_only_pages(files: Dict[str, str]) -> Dict[str, str]:
    """
    Deterministic post-processor for frontend-only generated pages.

    Fixes the 6 most common LLM mistakes without needing an LLM:

    1. Wrong chart prop `lines=` → `series=` (LineChart)
    2. Wrong chart prop `bars=`  → `series=` (BarChart)
    3. Wrong chart prop `xKey=`  → `xAxisKey=` (LineChart + BarChart)
    4. Wrong chart prop `donut=` → `innerRadius=` (PieChart)
    5. Chart direct sub-path import → barrel import via '../components/ui'
    6. `dataKey=` as a top-level LineChart/BarChart prop → wrapped in series

    Returns the modified files dict (same keys, values may be patched).
    """
    result: Dict[str, str] = {}
    for path, content in files.items():
        if not (path.startswith("frontend/src/pages/") and path.endswith(".tsx")):
            result[path] = content
            continue

        original = content

        # Fix 1 & 2: `lines={...}` → `series={...}` and `bars={...}` → `series={...}`
        content = re.sub(r'\blines=\{', 'series={', content)
        content = re.sub(r'\bbars=\{', 'series={', content)

        # Fix 3: `xKey=` → `xAxisKey=`
        content = re.sub(r'\bxKey=', 'xAxisKey=', content)

        # Fix 4: `donut={true}` / `donut` → `innerRadius={60}`
        content = re.sub(r'\bdonut=\{true\}', 'innerRadius={60}', content)
        content = re.sub(r'\bdonut=\{false\}', '', content)
        content = re.sub(r'\bdonut\b(?!=)', 'innerRadius={60}', content)

        # Fix 5: direct sub-path imports → barrel import
        # e.g. import LineChart from '../components/ui/charts/LineChart';
        #   or import { LineChart } from '../components/ui/charts/LineChart';
        def _replace_chart_direct_import(m: re.Match) -> str:
            import_stmt = m.group(0)
            # Extract component name
            name_match = re.search(r"(LineChart|BarChart|PieChart)", import_stmt)
            if not name_match:
                return import_stmt
            comp = name_match.group(1)
            return f"// [auto-fixed] chart import\n// Original: {import_stmt.strip()}"

        content = re.sub(
            r"import\s+(?:\{[^}]*\}|[A-Za-z]+)\s+from\s+'[^']*ui/charts/[^']*';?",
            _replace_chart_direct_import,
            content,
        )

        # Now ensure LineChart/BarChart/PieChart are in the barrel import if used
        used_charts = [c for c in ("LineChart", "BarChart", "PieChart") if f"<{c}" in content or f"{c}," in content or f"{c} " in content]
        if used_charts:
            # Check if there's already a barrel import from '../components/ui'
            barrel_match = re.search(
                r"(import\s*\{)([^}]*?)(\}\s*from\s*'\.\.\/components\/ui'\s*;?)",
                content,
            )
            if barrel_match:
                existing = barrel_match.group(2)
                existing_names = {n.strip() for n in existing.split(",") if n.strip()}
                to_add = [c for c in used_charts if c not in existing_names]
                if to_add:
                    new_imports = existing.rstrip() + (", " if existing.strip() else "") + ", ".join(to_add)
                    content = content[:barrel_match.start(2)] + new_imports + content[barrel_match.end(2):]
            else:
                # No barrel import yet — add one after the last 'react' import
                chart_import_line = f"import {{ {', '.join(used_charts)} }} from '../components/ui';\n"
                react_end = content.rfind("from 'react'")
                if react_end != -1:
                    insert_pos = content.find("\n", react_end) + 1
                    content = content[:insert_pos] + chart_import_line + content[insert_pos:]
                else:
                    content = chart_import_line + content

        if content != original:
            logger.info("[fix_frontend_only_pages] patched chart issues in %s", path)

        result[path] = content

    return result
