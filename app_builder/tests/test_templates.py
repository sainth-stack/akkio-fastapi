import sys
import os
import asyncio

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../app_builder/tests
parent_dir = os.path.dirname(os.path.dirname(current_dir)) # .../akkio-fastapi
sys.path.append(parent_dir)

from app_builder.services.template_service import detect_template, load_template
from app_builder.graph.builder_graph import app_builder_graph
from app_builder.schemas.requirements import UserRequirement
from app_builder.schemas.plan import ProjectPlan

def test_detection():
    print("Testing detection...")
    assert detect_template("Create a todo list app") == "todo-list"
    assert detect_template("I need a vacation budget planner") == "vacation-budget-planner"
    assert detect_template("inventory management") is None
    print("Detection test passed!")

def test_loading():
    print("Testing loading...")
    data = load_template("todo-list")
    assert data is not None
    assert data["app_name"] == "todo_app"
    
    data = load_template("vacation-budget-planner")
    assert data is not None
    assert data["app_name"] == "vacation_budget_planner"
    print("Loading test passed!")

async def test_graph_with_todo_template():
    print("Testing graph with template (todo list)...")
    initial_state = {
        "user_requirement": UserRequirement(description="Build a todo list"),
        "clarified_requirement": "",
        "plan": None,
        "architecture": None,
        "generated_files": None,
        "template_data": None,
        "error": ""
    }
    
    result = await app_builder_graph.ainvoke(initial_state)
    if result.get("error"):
        print(f"Graph error: {result['error']}")
    assert not result.get("error")
    assert result.get("template_data") is not None
    assert result["template_data"]["app_name"] == "todo_app"
    assert result.get("generated_files") is not None
    assert "backend/main.py" in result["generated_files"].files
    print("Graph test for todo-list passed!")

async def test_graph_with_vacation_template():
    print("Testing graph with template (vacation planner)...")
    initial_state = {
        "user_requirement": UserRequirement(description="vacation budget planner for Paris"),
        "clarified_requirement": "",
        "plan": None,
        "architecture": None,
        "generated_files": None,
        "template_data": None,
        "error": ""
    }
    
    result = await app_builder_graph.ainvoke(initial_state)
    assert not result.get("error")
    assert result.get("template_data") is not None
    assert result["template_data"]["app_name"] == "vacation_budget_planner"
    assert result.get("generated_files") is not None
    assert "backend/main.py" in result["generated_files"].files
    print("Graph test for vacation-budget-planner passed!")

async def test_graph_dynamic():
    print("Testing graph dynamic (Library Management system)...")
    initial_state = {
        "user_requirement": UserRequirement(description="Build a library management system to track books and members"),
        "clarified_requirement": "",
        "plan": None,
        "architecture": None,
        "generated_files": None,
        "template_data": None,
        "error": ""
    }
    
    result = await app_builder_graph.ainvoke(initial_state)
    if result.get("error"):
        print(f"Graph error: {result['error']}")
    assert not result.get("error")
    assert result.get("template_data") is None
    assert result.get("generated_files") is not None
    
    files = result["generated_files"].files
    assert "backend/main.py" in files
    assert "frontend/public/index.html" in files
    assert "frontend/src/index.js" in files
    
    # Check if domain-specific entities were inferred
    models_content = files.get("backend/models.py", "")
    print(f"DEBUG - entities inferred: {[f['name'] for f in result['architecture'].database_schema['tables']]}")
    print(f"DEBUG - models.py content:\n{models_content}")
    assert "class Book" in models_content or "class Member" in models_content
    assert "class Item" not in models_content or "class Book" in models_content
    
    print("Graph test for dynamic generation (schema-aware) passed!")

async def main():
    test_detection()
    test_loading()
    await test_graph_with_todo_template()
    await test_graph_with_vacation_template()
    await test_graph_dynamic()
    print("\nALL TEMPLATE TESTS PASSED!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
