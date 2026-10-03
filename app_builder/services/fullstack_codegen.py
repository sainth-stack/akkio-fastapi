"""Fullstack Builder codegen helpers: scaffold, theme, mock fallback, screen QA, deliverables."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("app_builder")

FULLSTACK_TEMPLATE_NAME = "base-fullstack-vite-mui"
TEMPLATES_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
FULLSTACK_DIR = os.path.join(TEMPLATES_BASE, FULLSTACK_TEMPLATE_NAME)

FULLSTACK_FROZEN = frozenset({
    "frontend/package.json",
    "frontend/vite.config.ts",
    "frontend/tsconfig.json",
    "frontend/index.html",
    "frontend/src/main.tsx",
    "frontend/src/api/client.ts",
    "frontend/src/vite-env.d.ts",
    "backend/database.py",
    "backend/requirements.txt",
    "backend/Dockerfile",
    "frontend/Dockerfile",
    "frontend/nginx.conf",
    "docker-compose.yml",
    ".env.example",
    "README.md",
})


def _read(rel_path: str) -> Optional[str]:
    full = os.path.join(FULLSTACK_DIR, rel_path)
    if not os.path.isfile(full):
        return None
    try:
        with open(full, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def get_fullstack_scaffold_files() -> Dict[str, str]:
    if not os.path.isdir(FULLSTACK_DIR):
        logger.error("[fullstack] template missing: %s", FULLSTACK_DIR)
        return {}
    files: Dict[str, str] = {}
    exclude_dirs = {"__pycache__", "node_modules", ".git", "dist", "build"}
    for root, dirs, filenames in os.walk(FULLSTACK_DIR):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        rel_root = os.path.relpath(root, FULLSTACK_DIR)
        for fn in filenames:
            if fn.endswith(".pyc"):
                continue
            if fn == "base.json":
                continue
            rel_path = os.path.join(rel_root, fn) if rel_root != "." else fn
            rel_path = rel_path.replace("\\", "/")
            content = _read(rel_path)
            if content is not None:
                files[rel_path] = content
    logger.info("[fullstack] loaded scaffold | files=%d", len(files))
    return files


def apply_theme_tokens(
    files: Dict[str, str],
    design_tokens: Optional[Dict[str, Any]],
    uiux: str = "",
) -> Dict[str, str]:
    """Regenerate MUI theme from design system tokens (full file, not regex patch)."""
    from app_builder.services.fullstack_frontend_generator import theme_from_tokens, build_theme_ts

    colors = theme_from_tokens(design_tokens, uiux=uiux)
    files["frontend/src/theme.ts"] = build_theme_ts(colors)
    return files


_DEFAULT_EXPORT_PATCH = "\n// Export as default so both import styles work\nexport default apiFetch;\n"

def ensure_mock_client(files: Dict[str, str]) -> Dict[str, str]:
    client = files.get("frontend/src/api/client.ts") or files.get("frontend/src/api/client.js") or ""
    if "mockFetch" not in client:
        scaffold = get_fullstack_scaffold_files()
        if "frontend/src/api/client.ts" in scaffold:
            files["frontend/src/api/client.ts"] = scaffold["frontend/src/api/client.ts"]
        if "frontend/src/api/mock.ts" in scaffold and "frontend/src/api/mock.ts" not in files:
            files["frontend/src/api/mock.ts"] = scaffold["frontend/src/api/mock.ts"]
    if "frontend/src/api/mock.ts" not in files:
        scaffold = get_fullstack_scaffold_files()
        if "frontend/src/api/mock.ts" in scaffold:
            files["frontend/src/api/mock.ts"] = scaffold["frontend/src/api/mock.ts"]

    # Ensure client.ts always has default export so LLM-generated pages compile
    # regardless of whether they use:  import apiFetch from '...'  OR  import { apiFetch } from '...'
    client_key = "frontend/src/api/client.ts"
    if client_key in files and "export default apiFetch" not in files[client_key]:
        files[client_key] = files[client_key].rstrip() + _DEFAULT_EXPORT_PATCH

    return files


def ensure_deliverables(files: Dict[str, str]) -> Dict[str, str]:
    scaffold = get_fullstack_scaffold_files()
    for path in ("docker-compose.yml", ".env.example", "README.md", "backend/seed.py", "backend/alembic.ini"):
        if path not in files and path in scaffold:
            files[path] = scaffold[path]
    return files


def merge_llm_into_fullstack(scaffold: Dict[str, str], llm_files: Dict[str, str]) -> Dict[str, str]:
    out = dict(scaffold)
    for path, content in (llm_files or {}).items():
        norm = path.replace("\\", "/").lstrip("/")
        if norm in FULLSTACK_FROZEN:
            continue
        if norm.endswith(".jsx") or norm.endswith("app.css"):
            continue
        if not content or not str(content).strip():
            continue
        if not isinstance(content, str):
            continue
        out[norm] = content
    return out


def _screen_names(architecture: Dict[str, Any], prd: str, uiux: str) -> List[str]:
    names: List[str] = []
    screens = architecture.get("screens") or []
    if isinstance(screens, list):
        for s in screens:
            if isinstance(s, dict) and s.get("name"):
                names.append(str(s["name"]))
            elif isinstance(s, str):
                names.append(s)
    comps = (architecture.get("frontend_structure") or {}).get("key_components") or []
    names.extend([str(c) for c in comps if c])
    blob = f"{prd}\n{uiux}"
    for m in re.finditer(r"(?im)^(?:#{1,3}\s+|\d+\.\s+)([A-Z][A-Za-z0-9 /&-]{2,40})$", blob):
        names.append(m.group(1).strip())
    seen = set()
    ordered = []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(n)
    return ordered[:40]


def run_screen_qa(
    files: Dict[str, str],
    architecture: Dict[str, Any],
    prd: str = "",
    uiux: str = "",
) -> Tuple[List[str], List[str]]:
    """Return (passed, missing) screen checks against generated frontend files."""
    frontend_blob = "\n".join(
        content for path, content in files.items() if path.startswith("frontend/")
    ).lower()
    passed: List[str] = []
    missing: List[str] = []
    for name in _screen_names(architecture, prd, uiux):
        tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if len(t) > 2]
        if not tokens:
            continue
        if all(t in frontend_blob for t in tokens[:2]):
            passed.append(name)
        else:
            missing.append(name)
    return passed, missing


def is_fullstack_generated_files(files: Dict[str, str]) -> bool:
    """True when codegen output is Agentic/fullstack (Vite+TS+MUI), not generic App.jsx shell."""
    if not isinstance(files, dict) or not files:
        return False
    pkg = files.get("frontend/package.json") or ""
    if "frontend/src/App.tsx" in files and "@mui/material" in pkg:
        return True
    if files.get("backend/risk_engine.py"):
        return True
    if "docker-compose.yml" in files and "frontend/tsconfig.json" in files:
        return True
    return False


def _fix_apifetch_imports(files: Dict[str, str]) -> Dict[str, str]:
    """
    Sweep all TypeScript files and fix LLM-generated wrong default imports of apiFetch.
    client.ts uses named export: export async function apiFetch(...)
    LLMs often write: import apiFetch from '../api/client'  ← wrong
    Should be:        import { apiFetch } from '../api/client'  ← correct
    """
    import re
    pattern = re.compile(
        r"import\s+apiFetch\s+from\s+['\"](\.\./)*api/client['\"]"
    )
    out = {}
    for path, content in files.items():
        if path.endswith((".ts", ".tsx")) and "api/client" in content:
            content = pattern.sub("import { apiFetch } from '../api/client'", content)
        out[path] = content
    return out


def post_process_fullstack_files(
    files: Dict[str, str],
    design_tokens: Optional[Dict[str, Any]] = None,
    uiux: str = "",
    requirement: str = "",
    prd: str = "",
) -> Dict[str, str]:
    """Post-process for fullstack apps — fills missing/stale pages, regenerates theme, never CRA-merges."""
    from api.app_creator.backend_verify_service import apply_deterministic_backend_fixes
    from app_builder.services.fullstack_app_generator import fill_missing_fullstack_files

    out = dict(files or {})
    # Fill stale or missing pages based on app mode (ecommerce / doc_chat / quality)
    if requirement or prd:
        try:
            out = fill_missing_fullstack_files(out, requirement, prd, uiux, design_tokens=design_tokens)
        except Exception as _fill_exc:
            logger.warning("[fullstack] fill_missing_fullstack_files failed: %s", _fill_exc)
    apply_deterministic_backend_fixes(out)
    out = apply_theme_tokens(out, design_tokens, uiux=uiux)
    out = ensure_mock_client(out)
    # Fix import mistakes LLMs make (default vs named imports for apiFetch)
    out = _fix_apifetch_imports(out)
    out.pop("frontend/src/App.jsx", None)
    out.pop("frontend/src/styles/app.css", None)
    return out


def apply_backend_failure_mock(files: Dict[str, str]) -> Dict[str, str]:
    """Keep frontend fully usable if backend verification failed."""
    files = ensure_mock_client(files)
    mock = files.get("frontend/src/api/mock.ts", "")
    if "MOCK MODE ENABLED" not in mock:
        files["frontend/src/api/mock.ts"] = (
            "// MOCK MODE ENABLED — backend verify failed; frontend uses fixtures.\n" + mock
        )
    return files
