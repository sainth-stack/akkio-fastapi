import os
from typing import Dict
from .runtime_paths import get_projects_dir

def read_project_files(project_name: str) -> Dict[str, str]:
    """
    Reads all text files in a project directory recursively.
    Returns a dict of {relative_path: content}.
    """
    project_path = os.path.join(get_projects_dir(), project_name)
    files_content = {}
    
    if not os.path.exists(project_path):
        return {}

    for root, _, files in os.walk(project_path):
        for file in files:
            # Skip common non-text or large files
            if file.startswith('.') or file.endswith(('.pyc', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.woff', '.woff2', '.ttf', '.eot', '.db', '.sqlite', '.sqlite3')):
                continue
            
            # Skip node_modules and venv/env directories if they are in the path
            rel_dir = os.path.relpath(root, project_path)
            if 'node_modules' in rel_dir.split(os.sep) or 'venv' in rel_dir.split(os.sep) or '.venv' in rel_dir.split(os.sep) or '__pycache__' in rel_dir.split(os.sep):
                continue

            full_path = os.path.join(root, file)
            try:
                 with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Store with relative path
                    rel_path = os.path.relpath(full_path, project_path)
                    # Normalize Windows paths
                    rel_path = rel_path.replace('\\', '/')
                    files_content[rel_path] = content
            except UnicodeDecodeError:
                # Skip binary files that slipped through
                pass
            except Exception as e:
                print(f"Error reading file {full_path}: {e}")

    return files_content
