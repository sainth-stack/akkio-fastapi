import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add akkio-fastapi to sys.path so we can import final_akio_apis
current_dir = os.path.dirname(os.path.abspath(__file__))
app_builder_dir = os.path.dirname(current_dir)
akkio_fastapi_dir = os.path.dirname(app_builder_dir)
sys.path.append(akkio_fastapi_dir)

from final_akio_apis import app

client = TestClient(app)

def test_generate_app_api():
    response = client.post(
        "/api/app-builder/generate",
        json={
            "requirement": "Create a task management app.",
            "project_name": "test_api_project"
        }
    )
    
    # Print error if failed
    if response.status_code != 200:
        print(f"API Error: {response.text}")

    assert response.status_code == 200
    # Endpoint streams NDJSON; parse it
    events = []
    for line in response.text.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(__import__("json").loads(line))

    plan = None
    files = None
    tree = None
    done = None
    for ev in events:
        if ev.get("plan"):
            plan = ev["plan"]
        if ev.get("files"):
            files = ev["files"]
        if ev.get("tree"):
            tree = ev["tree"]
        if ev.get("event") == "done":
            done = ev

    assert plan is not None and len(plan) > 0
    assert files is not None and len(files) > 0
    assert tree is not None and len(tree) > 0
    assert done is not None
    assert done.get("message") == "App generation successful"

    # Verify file persistence
    project_path = os.path.join(os.path.expanduser("~"), ".akkio", "app_builder", "runtime", "projects", "test_api_project")
    assert os.path.exists(project_path)

if __name__ == "__main__":
    test_generate_app_api()
