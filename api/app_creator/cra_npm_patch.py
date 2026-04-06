"""
Patch Create React App (react-scripts) package.json for Node 18+/20+ npm installs.

Nested webpack/schema-utils can pull incompatible ajv / ajv-keywords versions, causing:
  Error: Cannot find module 'ajv/dist/compile/codegen'

npm overrides force a single ajv@8 tree (see npm 8.3+ overrides).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

_AJV_OVERRIDE = "^8.12.0"


def _is_cra_package(pkg: Dict[str, Any]) -> bool:
    deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
    return "react-scripts" in deps


def patch_package_json_for_cra_ajv(frontend_dir: str) -> bool:
    """
    Merge npm ``overrides`` for ``ajv`` when react-scripts is present.
    Writes package.json if changed. Returns True if the file was modified.
    """
    pkg_path = os.path.join(frontend_dir, "package.json")
    if not os.path.isfile(pkg_path):
        return False
    try:
        with open(pkg_path, "r", encoding="utf-8") as f:
            pkg = json.load(f)
    except (json.JSONDecodeError, OSError, TypeError):
        return False
    if not isinstance(pkg, dict) or not _is_cra_package(pkg):
        return False

    overrides = dict(pkg.get("overrides") or {})
    cur = overrides.get("ajv")
    if isinstance(cur, str):
        major = cur.replace("^", "").replace("~", "").strip().split(".")[0]
        if major.isdigit() and int(major) >= 8:
            return False
    overrides["ajv"] = _AJV_OVERRIDE
    pkg["overrides"] = overrides

    try:
        with open(pkg_path, "w", encoding="utf-8") as f:
            json.dump(pkg, f, indent=2)
            f.write("\n")
    except OSError:
        return False
    return True
