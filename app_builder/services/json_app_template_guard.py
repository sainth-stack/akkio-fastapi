"""Keep json_app renderer files on the data-driven template (PageRenderer + data.json)."""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Tuple

from app_builder.services.frontend_only_template_guard import load_template_frontend_files

logger = logging.getLogger("app_builder")

# Never let codegen / build-fix LLM overwrite these — routes come from data.json.
_JSON_APP_PROTECTED: Tuple[str, ...] = (
    "frontend/src/App.tsx",
    "frontend/src/renderer/",
    "frontend/src/pages/PredictPage.tsx",
    "frontend/src/components/chat/",
    "frontend/src/components/predict/",
    "frontend/src/utils/ragDownload.ts",
)


def is_json_app_project(files: Dict[str, str]) -> bool:
    raw = files.get("frontend/src/mock/data.json")
    if not raw:
        return False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(data.get("pages"), list)
        and isinstance(data.get("navigation"), list)
        and data.get("version") is not None
    )


def _path_protected(path: str) -> bool:
    norm = path.replace("\\", "/").lstrip("/")
    for p in _JSON_APP_PROTECTED:
        if p.endswith("/"):
            if norm.startswith(p):
                return True
        elif norm == p:
            return True
    return False


def ensure_json_app_renderer(files: Dict[str, str]) -> Tuple[Dict[str, str], int]:
    """Restore PageRenderer-based App.tsx and related modules from template."""
    if not is_json_app_project(files):
        return files, 0
    template = load_template_frontend_files()
    if not template:
        return files, 0
    merged = dict(files)
    restored = 0
    for path, content in template.items():
        if _path_protected(path):
            merged[path] = content
            restored += 1
    if restored:
        logger.info("[json_app_guard] restored %d renderer file(s) from template", restored)
    return merged, restored


def repair_broken_legacy_app_tsx(files: Dict[str, str]) -> Tuple[Dict[str, str], bool]:
    """
    If App.tsx references *Page components (legacy FO router) but json_app data.json exists,
    replace App.tsx with the template json_app version.
    """
    if not is_json_app_project(files):
        return files, False
    app = files.get("frontend/src/App.tsx") or ""
    if "PageRenderer" in app and "ReportsPage" not in app:
        return files, False
    if re.search(r"<\w+Page\s*/?>", app) or "ReportsPage" in app:
        template = load_template_frontend_files()
        fixed = template.get("frontend/src/App.tsx")
        if fixed:
            out = dict(files)
            out["frontend/src/App.tsx"] = fixed
            logger.warning("[json_app_guard] replaced legacy App.tsx (missing page imports)")
            return out, True
    return files, False
