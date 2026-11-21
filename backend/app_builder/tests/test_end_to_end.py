import sys
import os
import shutil
import pytest

# Add akkio-fastapi to sys.path so we can import app_builder
current_dir = os.path.dirname(os.path.abspath(__file__))
app_builder_dir = os.path.dirname(current_dir)
akkio_fastapi_dir = os.path.dirname(app_builder_dir)
sys.path.append(akkio_fastapi_dir)

from app_builder.graph.builder_graph import app_builder_graph
from app_builder.schemas.requirements import UserRequirement
from app_builder.services.file_writer import file_writer

def test_end_to_end_flow():
    # Setup
    test_project_name = "test_inventory_app"
    test_req = "Create a simple inventory management app."
    
    # Cleanup previous run
    runtime_path = os.path.join(os.path.expanduser("~"), ".akkio", "app_builder", "runtime", "projects", test_project_name)
    if os.path.exists(runtime_path):
        shutil.rmtree(runtime_path)

    # Execution
    initial_state = {
        "user_requirement": UserRequirement(description=test_req),
        "clarified_requirement": "",
        "plan": None,
        "architecture": None,
        "generated_files": None,
        "error": ""
    }

    result = app_builder_graph.invoke(initial_state)

    # Assertions - Graph Execution
    assert result.get("error") == "", f"Graph execution failed: {result.get('error')}"
    assert result.get("generated_files") is not None
    assert len(result["generated_files"].files) > 0

    # Act - Write Files
    file_writer(test_project_name, result["generated_files"])

    # Assertions - File System
    assert os.path.exists(runtime_path)
    assert os.path.exists(os.path.join(runtime_path, "backend", "main.py"))
    assert os.path.exists(os.path.join(runtime_path, "frontend", "package.json"))
    assert os.path.exists(os.path.join(runtime_path, "README.md"))

    print(f"\nTest passed! Project generated at: {runtime_path}")

if __name__ == "__main__":
    test_end_to_end_flow()
