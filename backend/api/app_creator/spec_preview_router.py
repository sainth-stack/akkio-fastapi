"""Serve the shared JSON AppSpec renderer at /spec-app/{app_id}."""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from app_builder.services.json_renderer_build import ensure_json_renderer_built, renderer_dist_dir

logger = logging.getLogger("app_builder")

router = APIRouter(prefix="/spec-app", tags=["JSON Spec Preview"])


def _inject_spec_bootstrap(html: str, app_id: str, access_token: Optional[str]) -> str:
    spec_url = f"/api/app-builder/apps/{app_id}/spec"
    scripts = [
        f"window.__AKKIO_APP_ID__ = {json.dumps(app_id)};",
        f"window.__AKKIO_SPEC_URL__ = {json.dumps(spec_url)};",
        f"window.__AKKIO_BASE_PATH__ = {json.dumps(f'/spec-app/{app_id}')};",
    ]
    if access_token and access_token.strip():
        scripts.append(f"window.__AKKIO_ACCESS_TOKEN__ = {json.dumps(access_token.strip())};")
        scripts.append("""(function() {
  var token = window.__AKKIO_ACCESS_TOKEN__;
  if (!token || !window.fetch) return;
  var orig = window.fetch.bind(window);
  window.fetch = function(url, opts) {
    opts = opts || {};
    var headers = new Headers(opts.headers || {});
    if (!headers.has('Authorization')) headers.set('Authorization', 'Bearer ' + token);
    opts.headers = headers;
    return orig(url, opts);
  };
})();""")
    block = "<script>\n" + "\n".join(scripts) + "\n</script>"
    if "</head>" in html:
        return html.replace("</head>", block + "</head>", 1)
    return block + html


def _static_dir() -> str:
    try:
        return ensure_json_renderer_built()
    except Exception as exc:
        logger.exception("[spec_preview] renderer build failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"JSON renderer not ready: {exc}") from exc


@router.get("/assets/{path:path}")
async def spec_app_shared_assets(path: str):
    """Vite build uses base `/spec-app/` — assets load from /spec-app/assets/..."""
    static_dir = _static_dir()
    full = os.path.normpath(os.path.join(static_dir, "assets", path))
    root = os.path.abspath(static_dir)
    if not full.startswith(root):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(full)


@router.get("/{app_id}")
@router.get("/{app_id}/")
async def spec_app_root(app_id: str, access_token: Optional[str] = Query(None)):
    static_dir = _static_dir()
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, encoding="utf-8") as f:
        body = _inject_spec_bootstrap(f.read(), app_id, access_token)
    return HTMLResponse(content=body)


@router.get("/{app_id}/{path:path}")
async def spec_app_assets(app_id: str, path: str, access_token: Optional[str] = Query(None)):
    static_dir = _static_dir()
    full = os.path.normpath(os.path.join(static_dir, path))
    if not full.startswith(os.path.abspath(static_dir)):
        raise HTTPException(status_code=400, detail="Invalid path")
    if os.path.isfile(full):
        return FileResponse(full)
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, encoding="utf-8") as f:
        body = _inject_spec_bootstrap(f.read(), app_id, access_token)
    return HTMLResponse(content=body)
