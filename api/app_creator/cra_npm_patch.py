"""
Patch Create React App (react-scripts) for Node 18+/20+ builds.

fork-ts-checker-webpack-plugin + ajv/ajv-keywords version skew causes:
  TypeError: Cannot read properties of undefined (reading 'date')

Fix: @craco/craco removes ForkTsCheckerWebpackPlugin + npm overrides pin ajv@6.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, Union

CRA_NPM_OVERRIDES: Dict[str, Union[str, Dict[str, str]]] = {
    # schema-utils v4 (terser, css-minimizer, etc.) needs ajv v8 on Node 20+
    "ajv": "8.12.0",
    # fork-ts-checker still loads at webpack config import time — keep ajv v6 nested
    "fork-ts-checker-webpack-plugin": {
        "ajv": "6.12.6",
        "ajv-keywords": "3.5.2",
        "schema-utils": {
            "ajv": "6.12.6",
            "ajv-keywords": "3.5.2",
        },
    },
    "babel-loader": {
        "schema-utils": {
            "ajv": "6.12.6",
            "ajv-keywords": "3.5.2",
        },
    },
}

CRA_BUILD_ENV = {
    "GENERATE_SOURCEMAP": "false",
    "TSC_COMPILE_ON_ERROR": "true",
    "DISABLE_ESLINT_PLUGIN": "true",
    "SKIP_PREFLIGHT_CHECK": "true",
}

CRACO_CONFIG_JS = """/* Auto-generated — skip fork-ts-checker (ajv crash on Node 20+) */
module.exports = {
  webpack: {
    configure: (config) => {
      config.plugins = (config.plugins || []).filter(
        (p) => !p || !p.constructor || p.constructor.name !== 'ForkTsCheckerWebpackPlugin'
      );
      return config;
    },
  },
};
"""

_NODE_LEGACY = "NODE_OPTIONS=--openssl-legacy-provider "


def _is_cra_package(pkg: Dict[str, Any]) -> bool:
    deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
    return "react-scripts" in deps


def _uses_craco_scripts(scripts: dict) -> bool:
    for key in ("start", "build"):
        if "craco" not in str(scripts.get(key, "")):
            return False
    return True


def apply_cra_build_patch_to_package(pkg: Dict[str, Any]) -> bool:
    """Apply overrides, craco devDependency, and craco build scripts. Returns True if modified."""
    if not isinstance(pkg, dict) or not _is_cra_package(pkg):
        return False

    changed = False
    overrides = dict(pkg.get("overrides") or {})
    for key, val in CRA_NPM_OVERRIDES.items():
        if overrides.get(key) != val:
            overrides[key] = val
            changed = True
    if changed or not pkg.get("overrides"):
        pkg["overrides"] = overrides
        changed = True

    dev = dict(pkg.get("devDependencies") or {})
    if dev.get("@craco/craco") != "7.1.0":
        dev["@craco/craco"] = "7.1.0"
        changed = True
    # Drop legacy top-level ajv pins — overrides handle version skew
    for legacy in ("ajv", "ajv-keywords"):
        if legacy in dev:
            dev.pop(legacy, None)
            changed = True
    pkg["devDependencies"] = dev

    scripts = dict(pkg.get("scripts") or {})
    desired = {
        "start": _NODE_LEGACY + "craco start",
        "build": _NODE_LEGACY + "craco build",
    }
    for key, cmd in desired.items():
        if scripts.get(key) != cmd:
            scripts[key] = cmd
            changed = True
    pkg["scripts"] = scripts

    return changed


def _write_craco_config(frontend_dir: str) -> bool:
    path = os.path.join(frontend_dir, "craco.config.js")
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                if "ForkTsCheckerWebpackPlugin" in f.read():
                    return False
        with open(path, "w", encoding="utf-8") as f:
            f.write(CRACO_CONFIG_JS)
        return True
    except OSError:
        return False


def apply_cra_build_patch_on_disk(frontend_dir: str) -> bool:
    """Patch package.json + craco.config.js on disk. Returns True if anything changed."""
    changed = False
    pkg_path = os.path.join(frontend_dir, "package.json")
    if os.path.isfile(pkg_path):
        try:
            with open(pkg_path, "r", encoding="utf-8") as f:
                pkg = json.load(f)
            if apply_cra_build_patch_to_package(pkg):
                with open(pkg_path, "w", encoding="utf-8") as f:
                    json.dump(pkg, f, indent=2)
                    f.write("\n")
                changed = True
            elif not _uses_craco_scripts(pkg.get("scripts") or {}):
                apply_cra_build_patch_to_package(pkg)
                with open(pkg_path, "w", encoding="utf-8") as f:
                    json.dump(pkg, f, indent=2)
                    f.write("\n")
                changed = True
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    if _write_craco_config(frontend_dir):
        changed = True
    ensure_cra_build_env_on_disk(frontend_dir)
    return changed


# Backward-compatible names used elsewhere
def apply_cra_overrides_to_package(pkg: Dict[str, Any]) -> bool:
    return apply_cra_build_patch_to_package(pkg)


def patch_package_json_for_cra_ajv(frontend_dir: str) -> bool:
    return apply_cra_build_patch_on_disk(frontend_dir)


def apply_cra_build_patch_in_files(files: Dict[str, str]) -> None:
    """Patch in-memory generated files before writing to disk."""
    key = "frontend/package.json"
    if key in files:
        try:
            pkg = json.loads(files[key])
            if apply_cra_build_patch_to_package(pkg):
                files[key] = json.dumps(pkg, indent=2)
        except (json.JSONDecodeError, TypeError):
            pass
    files["frontend/craco.config.js"] = CRACO_CONFIG_JS
    ensure_cra_build_env_file(files)


def patch_package_json_in_files(files: Dict[str, str]) -> None:
    apply_cra_build_patch_in_files(files)


def ensure_cra_build_env_file(files: Dict[str, str]) -> None:
    key = "frontend/.env"
    existing = (files.get(key) or "").strip()
    lines_out = [l for l in existing.splitlines() if l.strip()] if existing else []
    for k, v in CRA_BUILD_ENV.items():
        if not any(l.startswith(k + "=") for l in lines_out):
            lines_out.append(f"{k}={v}")
    files[key] = "\n".join(lines_out) + "\n"


def ensure_cra_build_env_on_disk(frontend_dir: str) -> bool:
    env_path = os.path.join(frontend_dir, ".env")
    existing: Dict[str, str] = {}
    if os.path.isfile(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f.read().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip()
        except OSError:
            pass
    changed = any(existing.get(k) != v for k, v in CRA_BUILD_ENV.items())
    if not changed and os.path.isfile(env_path):
        return False
    try:
        merged = {**existing, **CRA_BUILD_ENV}
        with open(env_path, "w", encoding="utf-8") as f:
            for k, v in merged.items():
                f.write(f"{k}={v}\n")
    except OSError:
        return False
    return True


def clear_frontend_node_modules(frontend_dir: str) -> None:
    node_modules = os.path.join(frontend_dir, "node_modules")
    lockfile = os.path.join(frontend_dir, "package-lock.json")
    if os.path.isdir(node_modules):
        shutil.rmtree(node_modules, ignore_errors=True)
    if os.path.isfile(lockfile):
        try:
            os.remove(lockfile)
        except OSError:
            pass


def prepare_frontend_dir_on_disk(frontend_dir: str) -> bool:
    """
    Always apply CRA/craco patch and force a clean npm install.
    Returns True (caller should always run npm install).
    """
    apply_cra_build_patch_on_disk(frontend_dir)
    clear_frontend_node_modules(frontend_dir)
    return True
