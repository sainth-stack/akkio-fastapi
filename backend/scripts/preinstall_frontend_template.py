#!/usr/bin/env python3
"""
Pre-install node_modules for base-frontend-vite-mui once, then reuse via
copy or symlink so subsequent app builds skip a full `npm install`.

Usage:
    python scripts/preinstall_frontend_template.py          # install + cache
    python scripts/preinstall_frontend_template.py --check  # exit 0 if cached

Cache location: ~/.akkio/app_builder/nm_cache/base-frontend-vite-mui/node_modules/

How builds use the cache (vite_build.py integration):
    1. Check if cache_dir/node_modules exists and is non-empty.
    2. If yes: os.symlink(cache_dir/node_modules, project/frontend/node_modules)
    3. If no : run `npm install` normally, then copy cache for next time.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_FASTAPI_DIR = _SCRIPT_DIR.parent
_TEMPLATE_FRONTEND = _FASTAPI_DIR / "app_builder" / "templates" / "base-frontend-vite-mui" / "frontend"
_CACHE_ROOT = Path.home() / ".akkio" / "app_builder" / "nm_cache"
_CACHE_DIR = _CACHE_ROOT / "base-frontend-vite-mui"
_CACHED_NM = _CACHE_DIR / "node_modules"


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"[preinstall] $ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def is_cached() -> bool:
    """Return True if the cache already contains a non-empty node_modules."""
    return _CACHED_NM.is_dir() and any(_CACHED_NM.iterdir())


def install(force: bool = False) -> bool:
    """
    Run `npm install` inside the template frontend directory, then copy
    the resulting node_modules to the cache.

    Args:
        force: Re-install even if cache already exists.

    Returns:
        True on success.
    """
    if is_cached() and not force:
        print(f"[preinstall] Cache already exists at {_CACHED_NM} — skipping install.")
        print("[preinstall] Use --force to reinstall.")
        return True

    if not _TEMPLATE_FRONTEND.is_dir():
        print(f"[preinstall] ERROR: template not found at {_TEMPLATE_FRONTEND}", file=sys.stderr)
        return False

    # 1. npm install inside the template
    rc = _run(["npm", "install", "--prefer-offline", "--no-fund", "--no-audit"], _TEMPLATE_FRONTEND)
    if rc != 0:
        print(f"[preinstall] npm install failed (exit {rc})", file=sys.stderr)
        return False

    # 2. Copy node_modules to cache
    src_nm = _TEMPLATE_FRONTEND / "node_modules"
    if not src_nm.is_dir():
        print("[preinstall] ERROR: node_modules not found after npm install", file=sys.stderr)
        return False

    print(f"[preinstall] Caching node_modules → {_CACHED_NM} …", flush=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _CACHED_NM.exists():
        shutil.rmtree(_CACHED_NM)

    shutil.copytree(src_nm, _CACHED_NM, symlinks=True)
    print(f"[preinstall] ✓ Cached {sum(1 for _ in _CACHED_NM.rglob('*'))} entries")
    return True


def link_into_project(project_frontend_dir: str | Path) -> bool:
    """
    Symlink the cached node_modules into a generated project's frontend dir.

    Call this from vite_build.py *before* running `vite build`:

        from scripts.preinstall_frontend_template import link_into_project
        if not link_into_project(project_frontend):
            subprocess.run(["npm", "install"], cwd=project_frontend, check=True)

    Args:
        project_frontend_dir: Absolute path to the generated project's frontend/.

    Returns:
        True if the symlink was created successfully, False if cache is missing.
    """
    project_frontend_dir = Path(project_frontend_dir)
    target_nm = project_frontend_dir / "node_modules"

    if not is_cached():
        return False

    # If node_modules already exists (from a previous build), skip
    if target_nm.exists() or target_nm.is_symlink():
        return True

    try:
        os.symlink(_CACHED_NM, target_nm)
        print(f"[preinstall] Symlinked node_modules into {project_frontend_dir}")
        return True
    except OSError as exc:
        print(f"[preinstall] Symlink failed ({exc}) — falling back to npm install", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-install and cache node_modules for base-frontend-vite-mui."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 0 if cache exists, 1 otherwise (no install).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-install even if cache already exists.",
    )
    args = parser.parse_args()

    if args.check:
        if is_cached():
            print(f"[preinstall] Cache exists at {_CACHED_NM}")
            sys.exit(0)
        else:
            print("[preinstall] Cache NOT found")
            sys.exit(1)

    ok = install(force=args.force)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
