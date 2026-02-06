import sys
import os

# Setup path to modules
# We assume we are running this script using: python akkio-fastapi/app_builder/main.py
# So we need to ensure we can import 'app_builder' if we are treating it as a package
# We append the parent directory of 'app_builder' which is 'akkio-fastapi'? 
# No, if imports are `from app_builder...` then we need the parent of `app_builder` in sys.path.
# If cwd is `.../Akkio`, then `akkio-fastapi` is a folder.
# We probably want to add `akkio-fastapi` to sys.path.

current_dir = os.path.dirname(os.path.abspath(__file__)) # .../app_builder
parent_dir = os.path.dirname(current_dir) # .../akkio-fastapi
sys.path.append(parent_dir)

from app_builder.graph.builder_graph import app_builder_graph
from app_builder.schemas.requirements import UserRequirement
from app_builder.services.file_writer import file_writer

def main():
    print(">>> Starting Day 1 App Builder Flow")
    
    input_req = "Create a simple inventory management app to track items with CRUD functionality."
    print(f">>> User Requirement: {input_req}")

    initial_state = {
        "user_requirement": UserRequirement(description=input_req),
        "clarified_requirement": "",
        "plan": None,
        "architecture": None,
        "generated_files": None,
        "error": ""
    }

    print(">>> Invoking LangGraph...")
    try:
        result = app_builder_graph.invoke(initial_state)
    except Exception as e:
        print(f">>> Graph Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    if result.get("error"):
        print(f">>> Error in Graph: {result['error']}")
        return

    generated_files = result.get("generated_files")
    if generated_files and generated_files.files:
        print(f">>> Generated {len(generated_files.files)} files.")
        file_writer("inventory_app", generated_files)
        print(">>> SUCCESS: Project generated in runtime projects directory")
    else:
        print(">>> Warning: No files generated.")

if __name__ == "__main__":
    main()
