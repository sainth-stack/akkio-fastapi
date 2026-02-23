import os
import json
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from app_builder.services.file_writer import write_project_file

async def generate_tests(
    project_name: str,
    backend_files: Dict[str, str],
    llm
) -> Dict[str, str]:
    """
    Generates pytest scripts for the backend based on the existing code.
    focuses on main.py, models.py, and schemas.py.
    """
    
    # Extract relevant file content for context
    main_py = backend_files.get("backend/main.py", "")
    models_py = backend_files.get("backend/models.py", "")
    schemas_py = backend_files.get("backend/schemas.py", "")
    
    if not main_py:
        return {}

    system_prompt = """You are an expert QA automation engineer specializing in FastAPI and pytest.
Your task is to generate a comprehensive test suite for a FastAPI application.

**REQUIREMENTS:**
1. Use `pytest` and `fastapi.testclient`.
2. Generate a single file `backend/tests/test_main.py`.
3. Include tests for:
   - Health check / root endpoint (if available)
   - CRUD operations for all entities defined in `main.py` / `routes`.
   - Edge cases (e.g., 404 Not Found, 422 Validation Error).
4. Use a fixture for `TestClient` if needed, or instantiate it globally.
5. The test file MUST be self-contained and runnable from the `backend` directory via `pytest`.
6. Assume `main.py` creates the `app` object.
7. Mocking is NOT required for SQLite; use a temporary in-memory SQLite db or the existing one. 
   **PREFERENCE**: Use `TestClient(app)` directly. If the app uses a database dependency, override it to use an in-memory SQLite database for isolation.

**OUTPUT FORMAT:**
Return ONLY the python code for the test file. No markdown formatting.
"""

    user_prompt = f"""Generate `backend/tests/test_main.py` for this FastAPI app:

**backend/main.py**:
{main_py}

**backend/models.py**:
{models_py}

**backend/schemas.py**:
{schemas_py}

Generate the full `test_main.py` content.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = await llm.ainvoke(messages)
    content = response.content if hasattr(response, 'content') else str(response)
    
    # Strip potential markdown code blocks
    if content.startswith("```python"):
        content = content[9:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
        
    test_file_path = "backend/tests/test_main.py"
    
    # Write to disk
    write_project_file(project_name, test_file_path, content.strip())
    
    return {test_file_path: content.strip()}
