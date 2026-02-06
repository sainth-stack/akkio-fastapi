import os
from ..schemas.files import GeneratedFiles
from .runtime_paths import get_projects_dir

def file_writer(project_name: str, generated_files: GeneratedFiles):
    """
    Writes generated files to disk.
    """
    base_path = os.path.join(get_projects_dir(), project_name)
    
    
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    
    print(f"Writing to: {base_path}")

    for relative_path, content in generated_files.files.items():
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
    
    print(f"- Written: {relative_path}")

def write_project_file(project_name: str, relative_path: str, content: str):
    """Writes a single file for a project, ensuring directories exist."""
    base_path = os.path.join(get_projects_dir(), project_name)
    
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        
    write_single_file_to_disk(base_path, relative_path, content)
