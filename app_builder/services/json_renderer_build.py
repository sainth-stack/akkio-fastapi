"""Build the shared JSON spec renderer (once, cached)."""
from __future__ import annotations

import logging
import os
import subprocess

from api.app_creator.pipeline_helpers import npm_build_timeout, npm_install_timeout

logger = logging.getLogger("app_builder")

_TEMPLATE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "templates",
    "base-json-renderer",
    "frontend",
)
_CACHE_DIST = os.path.join(
    os.path.expanduser("~"),
    ".akkio",
    "app_builder",
    "json_renderer_dist",
)


def renderer_dist_dir() -> str:
    return _CACHE_DIST


def ensure_json_renderer_built() -> str:
    """
    npm install + vite build for base-json-renderer with VITE_BASE_PATH=/spec-app/.
    Returns path to dist/ or raises RuntimeError.
    """
    dist = _CACHE_DIST
    index = os.path.join(dist, "index.html")
    if os.path.isfile(index):
        return dist

    if not os.path.isdir(_TEMPLATE_ROOT):
        raise RuntimeError(f"JSON renderer template missing: {_TEMPLATE_ROOT}")

    logger.info("[json_renderer] building shared renderer (first run)...")
    subprocess.run(
        ["npm", "install", "--legacy-peer-deps"],
        cwd=_TEMPLATE_ROOT,
        check=True,
        timeout=npm_install_timeout(),
        capture_output=True,
    )
    env = os.environ.copy()
    env["VITE_BASE_PATH"] = "/spec-app/"
    subprocess.run(
        ["npm", "run", "build"],
        cwd=_TEMPLATE_ROOT,
        check=True,
        timeout=npm_build_timeout(),
        env=env,
        capture_output=True,
    )
    built = os.path.join(_TEMPLATE_ROOT, "dist")
    if not os.path.isfile(os.path.join(built, "index.html")):
        raise RuntimeError("JSON renderer build did not produce dist/index.html")

    os.makedirs(os.path.dirname(_CACHE_DIST), exist_ok=True)
    if os.path.isdir(dist):
        import shutil
        shutil.rmtree(dist, ignore_errors=True)
    import shutil
    shutil.copytree(built, dist)
    logger.info("[json_renderer] cached dist at %s", dist)
    return dist
