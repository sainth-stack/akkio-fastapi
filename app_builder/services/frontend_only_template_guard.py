"""Keep base-frontend-vite-mui UI kit immutable — fixes common barrel export breakages."""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger("app_builder")

TEMPLATE_NAME = "base-frontend-vite-mui"
_UI_KIT_PREFIX = "frontend/src/components/ui/"

_FROZEN_PATHS = frozenset({
    "frontend/src/components/ui/index.ts",
    "frontend/src/main.tsx",
    "frontend/vite.config.ts",
    "frontend/tsconfig.json",
    "frontend/index.html",
    "frontend/src/vite-env.d.ts",
    "frontend/src/theme/index.ts",
})

_BARREL_CHART_FIXES = (
    (re.compile(r"from\s+['\"]\./LineChart['\"]"), "from './charts/LineChart'"),
    (re.compile(r"from\s+['\"]\./BarChart['\"]"), "from './charts/BarChart'"),
    (re.compile(r"from\s+['\"]\./PieChart['\"]"), "from './charts/PieChart'"),
)


def _templates_dir() -> str:
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "templates",
    )
    return os.path.join(base, TEMPLATE_NAME)


def load_template_frontend_files() -> Dict[str, str]:
    """Load frontend/** from base-frontend-vite-mui (no node_modules/dist)."""
    root = _templates_dir()
    if not os.path.isdir(root):
        logger.error("[fo_template_guard] template missing: %s", root)
        return {}
    out: Dict[str, str] = {}
    exclude = {"__pycache__", "node_modules", ".git", "dist", "build"}
    for walk_root, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if d not in exclude]
        rel_root = os.path.relpath(walk_root, root)
        for fn in filenames:
            if fn.endswith(".pyc") or fn == "base.json":
                continue
            rel = os.path.join(rel_root, fn) if rel_root != "." else fn
            rel = rel.replace("\\", "/")
            if not rel.startswith("frontend/"):
                continue
            try:
                with open(os.path.join(walk_root, fn), encoding="utf-8") as fh:
                    out[rel] = fh.read()
            except OSError:
                pass
    return out


def fix_ui_barrel_index(content: str) -> str:
    """Repair LLM-broken chart re-exports in components/ui/index.ts."""
    if not content:
        return content
    fixed = content
    for pattern, repl in _BARREL_CHART_FIXES:
        fixed = pattern.sub(repl, fixed)
    return fixed


def is_ui_kit_path(path: str) -> bool:
    norm = path.replace("\\", "/").lstrip("/")
    return norm in _FROZEN_PATHS or norm.startswith(_UI_KIT_PREFIX)


def restore_template_ui_kit(files: Dict[str, str]) -> Tuple[Dict[str, str], int]:
    """
    Overlay immutable template files (full UI kit + frozen infra) onto generated files.
    Returns (merged_files, restored_file_count).
    """
    template = load_template_frontend_files()
    if not template:
        return files, 0

    merged = dict(files)
    restored = 0
    for path, content in template.items():
        if path in _FROZEN_PATHS or path.startswith(_UI_KIT_PREFIX):
            merged[path] = content
            restored += 1

    barrel = "frontend/src/components/ui/index.ts"
    if barrel in merged:
        patched = fix_ui_barrel_index(merged[barrel])
        if patched != merged[barrel]:
            merged[barrel] = patched
            restored += 1

    return merged, restored


def sync_files_to_project_disk(project_name: str, files: Dict[str, str], paths: Tuple[str, ...]) -> None:
    """Write selected paths to the project runtime directory."""
    if not paths:
        return
    from app_builder.services.file_writer import write_single_file_to_disk
    from app_builder.services.runtime_paths import project_write_root

    root = project_write_root(project_name)
    for rel in paths:
        if rel in files:
            write_single_file_to_disk(root, rel, files[rel])


def is_frontend_only_project(files: Dict[str, str]) -> bool:
    """True for base-frontend-vite-mui style apps (UI kit present, no generated backend)."""
    keys = {k.replace("\\", "/") for k in files}
    has_kit = any(k.startswith(_UI_KIT_PREFIX) for k in keys)
    has_backend = "backend/main.py" in keys
    return has_kit and not has_backend


def ensure_frontend_only_kit(
    files: Dict[str, str],
    project_name: Optional[str] = None,
) -> Tuple[Dict[str, str], int]:
    """Restore UI kit in memory; optionally sync to project disk."""
    if not is_frontend_only_project(files):
        return files, 0
    merged, count = restore_template_ui_kit(files)
    if project_name and count:
        paths = tuple(p for p in merged if is_ui_kit_path(p) or p in _FROZEN_PATHS)
        sync_files_to_project_disk(project_name, merged, paths)
    return merged, count


def restore_and_sync_ui_kit(project_name: str, files: Dict[str, str]) -> Tuple[Dict[str, str], int]:
    """Restore kit in memory and write UI kit + frozen paths to disk before vite build."""
    merged, count = restore_template_ui_kit(files)
    if count:
        paths = tuple(
            p for p in merged
            if is_ui_kit_path(p) or p in _FROZEN_PATHS
        )
        sync_files_to_project_disk(project_name, merged, paths)
        logger.info(
            "[fo_template_guard] restored %d template file(s) from %s | project=%s",
            count,
            TEMPLATE_NAME,
            project_name,
        )
    return merged, count
