"""
Serve built frontends for generated apps at /app/{project_id}.
SPA fallback: unknown paths return index.html.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from app_builder.services.runtime_paths import resolve_project_root

logger = logging.getLogger("app_builder")

router = APIRouter(prefix="/app", tags=["App Static"])


def _resolve_static_dir(project_id: str) -> Optional[str]:
    """Resolve directory containing built frontend (dist/ or build/)."""
    project_root = resolve_project_root(project_id)
    if not os.path.isdir(project_root):
        return None
    # Check frontend/dist
    frontend_dist = os.path.join(project_root, "frontend", "dist")
    if os.path.isdir(frontend_dist):
        return frontend_dist
    # Check frontend/client/dist
    client_dist = os.path.join(project_root, "frontend", "client", "dist")
    if os.path.isdir(client_dist):
        return client_dist
    # Check frontend/client/build (CRA nested client)
    client_build = os.path.join(project_root, "frontend", "client", "build")
    if os.path.isdir(client_build):
        return client_build
    # Check root dist (single frontend app)
    root_dist = os.path.join(project_root, "dist")
    if os.path.isdir(root_dist):
        return root_dist
    # CRA uses build/
    frontend_build = os.path.join(project_root, "frontend", "build")
    if os.path.isdir(frontend_build):
        return frontend_build
    root_build = os.path.join(project_root, "build")
    if os.path.isdir(root_build):
        return root_build
    return None


def _inject_preview_auth(html_content: str, access_token: str | None, project_id: str | None = None) -> str:
    """Inject JWT and API base for preview iframe."""
    scripts = []
    if project_id:
        api_base = f"/api/apps/{project_id}"
        scripts.append(f"window.__AKKIO_API_BASE__ = {json.dumps(api_base)};")
    if access_token and access_token.strip():
        token_js = json.dumps(access_token.strip())
        scripts.append(f"window.__AKKIO_ACCESS_TOKEN__ = {token_js};")
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
    if not scripts:
        return html_content
    script = "<script>\n" + "\n".join(scripts) + "\n</script>"
    if "</head>" in html_content:
        return html_content.replace("</head>", script + "</head>", 1)
    return script + html_content


def _serve_index(static_dir: str, access_token: str | None = None, project_id: str | None = None):
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    if access_token or project_id:
        with open(index_path, "r", encoding="utf-8", errors="replace") as f:
            body = _inject_preview_auth(f.read(), access_token, project_id)
        return HTMLResponse(content=body)
    return FileResponse(index_path)


@router.get("/{project_id}")
@router.get("/{project_id}/")
async def serve_app_root(project_id: str, access_token: Optional[str] = Query(None)):
    """Serve index.html for /app/{project_id}."""
    static_dir = _resolve_static_dir(project_id)
    if not static_dir:
        raise HTTPException(status_code=404, detail="App not found or not built. Run build first.")
    return _serve_index(static_dir, access_token, project_id)


@router.get("/{project_id}/{path:path}")
async def serve_app_path(
    project_id: str,
    path: str,
    access_token: Optional[str] = Query(None),
):
    """Serve static file or index.html for SPA routing."""
    static_dir = _resolve_static_dir(project_id)
    if not static_dir:
        raise HTTPException(status_code=404, detail="App not found or not built")
    full_path = os.path.normpath(os.path.join(static_dir, path))
    if not full_path.startswith(os.path.abspath(static_dir)):
        raise HTTPException(status_code=400, detail="Invalid path")
    if os.path.isfile(full_path):
        return FileResponse(full_path)
    # SPA fallback
    return _serve_index(static_dir, access_token, project_id)
