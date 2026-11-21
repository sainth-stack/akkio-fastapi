"""Vite frontend build helpers (replaces CRA/craco for new generated apps)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Optional, Tuple

from api.app_creator.pipeline_helpers import npm_build_timeout, npm_install_timeout


def is_vite_frontend(frontend_dir: str) -> bool:
    pkg = os.path.join(frontend_dir, "package.json")
    if not os.path.isfile(pkg):
        return False
    try:
        with open(pkg, "r", encoding="utf-8") as f:
            data = json.load(f)
        scripts = json.dumps(data.get("scripts") or {})
        dev = json.dumps(data.get("devDependencies") or {})
        return "vite" in scripts.lower() or "vite" in dev.lower()
    except (json.JSONDecodeError, OSError):
        return os.path.isfile(os.path.join(frontend_dir, "vite.config.js"))


def prepare_vite_frontend_dir(frontend_dir: str) -> None:
    """Clean install for reproducible Vite builds."""
    for name in ("node_modules",):
        path = os.path.join(frontend_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    lock = os.path.join(frontend_dir, "package-lock.json")
    if os.path.isfile(lock):
        try:
            os.remove(lock)
        except OSError:
            pass


def run_vite_build(
    frontend_dir: str,
    project_name: str,
    install: bool = True,
) -> Tuple[Optional[str], Optional[str], str]:
    logs: list[str] = []
    try:
        if install:
            logs.append("[vite] npm install --legacy-peer-deps")
            r = subprocess.run(
                ["npm", "install", "--legacy-peer-deps"],
                cwd=frontend_dir,
                capture_output=True,
                timeout=npm_install_timeout(),
            )
            out = (r.stdout or b"").decode("utf-8", errors="replace")
            out += (r.stderr or b"").decode("utf-8", errors="replace")
            logs.append(out[-4000:])
            if r.returncode != 0:
                return None, f"npm install failed: {out[-800:]}", "\n".join(logs)

        build_env = os.environ.copy()
        build_env["VITE_BASE_PATH"] = f"/app/{project_name}/"

        logs.append(f"[vite] VITE_BASE_PATH={build_env['VITE_BASE_PATH']} npm run build")
        r = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            capture_output=True,
            timeout=npm_build_timeout(),
            env=build_env,
        )
        out = (r.stdout or b"").decode("utf-8", errors="replace")
        out += (r.stderr or b"").decode("utf-8", errors="replace")
        logs.append(out[-8000:])
        if r.returncode != 0:
            return None, f"vite build failed: {out[-1200:]}", "\n".join(logs)

        dist = os.path.join(frontend_dir, "dist")
        if os.path.isdir(dist):
            return dist, None, "\n".join(logs)
        return None, "vite build completed but dist/ not found", "\n".join(logs)
    except subprocess.TimeoutExpired:
        return None, "Vite build timed out", "\n".join(logs)
    except FileNotFoundError:
        return None, "npm not found", "\n".join(logs)
    except Exception as exc:
        return None, str(exc), "\n".join(logs)
