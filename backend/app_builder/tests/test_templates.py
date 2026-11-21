import sys
import os
import asyncio

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from app_builder.services.template_service import detect_template, load_template, get_template_code_files
from app_builder.graph.builder_graph import app_builder_graph
from app_builder.schemas.requirements import UserRequirement


def test_detection():
    """Two templates: ideas-generator (LLM) and todo-list (normal)."""
    print("Testing detection...")
    assert detect_template("Create a todo list app") == "todo-list"
    assert detect_template("I need a vacation budget planner") == "ideas-generator"
    assert detect_template("Create a LinkedIn post generator") == "ideas-generator"
    assert detect_template("ideas generator for startups") == "ideas-generator"
    assert detect_template("language translator") == "ideas-generator"
    assert detect_template("inventory management") is None
    print("Detection test passed!")


def test_loading():
    """Load config for both templates."""
    print("Testing loading...")
    data = load_template("todo-list")
    assert data is not None
    assert data["app_name"] == "todo_app"

    data = load_template("ideas-generator")
    assert data is not None
    assert data["app_name"] == "ideas_generator"
    print("Loading test passed!")


async def test_graph_with_todo_template():
    """Todo list uses todo-list template."""
    print("Testing graph with template (todo list)...")
    initial_state = {"user_requirement": UserRequirement(description="Build a todo list")}

    result = await app_builder_graph.ainvoke(initial_state)
    if result.get("error"):
        print(f"Graph error: {result['error']}")
    assert not result.get("error")
    assert result.get("template_name") == "todo-list"
    gen_files = result.get("generated_files")
    assert gen_files is not None
    files = gen_files if isinstance(gen_files, dict) else getattr(gen_files, "files", gen_files)
    assert "backend/main.py" in files
    print("Graph test for todo-list passed!")


async def test_graph_with_ideas_template():
    """Travel/vacation/LinkedIn uses ideas-generator (LLM) template."""
    print("Testing graph with template (ideas-generator - vacation)...")
    initial_state = {"user_requirement": UserRequirement(description="vacation budget planner for Paris")}

    result = await app_builder_graph.ainvoke(initial_state)
    assert not result.get("error")
    assert result.get("template_name") == "ideas-generator"
    gen_files = result.get("generated_files")
    assert gen_files is not None
    files = gen_files if isinstance(gen_files, dict) else getattr(gen_files, "files", gen_files)
    assert "backend/main.py" in files
    assert "frontend/src/App.js" in files
    print("Graph test for ideas-generator (vacation) passed!")


def test_template_files():
    """Both templates have required files."""
    print("Testing template files...")
    todo_files = get_template_code_files("todo-list")
    assert todo_files
    assert "backend/main.py" in todo_files
    assert "frontend/src/App.js" in todo_files

    ideas_files = get_template_code_files("ideas-generator")
    assert ideas_files
    assert "backend/main.py" in ideas_files
    assert "frontend/src/App.js" in ideas_files
    assert "gen_type" in ideas_files.get("backend/main.py", "")
    print("Template files test passed!")


async def main():
    test_detection()
    test_loading()
    test_template_files()
    await test_graph_with_todo_template()
    await test_graph_with_ideas_template()
    # Skip test_graph_dynamic - requires full LLM for library management (no template match)
    print("\nALL TEMPLATE TESTS PASSED!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
