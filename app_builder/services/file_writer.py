import logging
import os
from ..schemas.files import GeneratedFiles
from .runtime_paths import project_write_root

logger = logging.getLogger("app_builder")

def _should_skip_path(rel_path: str) -> bool:
    """Skip venv, node_modules, and other generated/dependency paths."""
    parts = rel_path.replace("\\", "/").lower().split("/")
    if ".venv" in parts or "venv" in parts or "node_modules" in parts or "site-packages" in parts:
        return True
    return False


def file_writer(project_name: str, generated_files: GeneratedFiles):
    """
    Writes generated files to disk.
    """
    base_path = project_write_root(project_name)
    files = {k: v for k, v in generated_files.files.items() if not _should_skip_path(k)}
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    logger.info("[file_writer] writing %d files to project=%s | path=%s", len(files), project_name, base_path)
    for relative_path, content in files.items():
        write_single_file_to_disk(base_path, relative_path, content)

def write_single_file_to_disk(base_path: str, relative_path: str, content: str):
    """Helper to write a single file, handling directories safely."""
    # Skip if path seems to be a directory
    if relative_path.endswith("/") or relative_path.endswith("\\"):
        print(f"Skipping directory path: {relative_path}")
        return

    full_path = os.path.join(base_path, relative_path)
    dir_name = os.path.dirname(full_path)
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    
    # Check if it's actually a directory (if it exists)
    if os.path.isdir(full_path):
        print(f"Skipping writing to directory: {full_path}")
        return

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

def write_project_file(project_name: str, relative_path: str, content: str) -> bool:
    """Writes a single file for a project. Returns False if path was skipped (venv etc.)."""
    if _should_skip_path(relative_path):
        logger.debug("[file_writer] skipping venv/site-packages path: %s", relative_path)
        return False
    base_path = project_write_root(project_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    write_single_file_to_disk(base_path, relative_path, content)
    return True
