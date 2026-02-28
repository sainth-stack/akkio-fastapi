"""
Serve built frontends for generated apps at /app/{project_id}.
SPA fallback: unknown paths return index.html.
"""
import os
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

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


@router.get("/{project_id}")
@router.get("/{project_id}/")
async def serve_app_root(project_id: str):
    """Serve index.html for /app/{project_id}."""
    static_dir = _resolve_static_dir(project_id)
    if not static_dir:
        raise HTTPException(status_code=404, detail="App not found or not built. Run build first.")
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@router.get("/{project_id}/{path:path}")
async def serve_app_path(project_id: str, path: str):
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
    index_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Not found")
