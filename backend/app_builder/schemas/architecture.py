from pydantic import BaseModel
from typing import Dict, Any

class ArchitectureDecision(BaseModel):
    frontend_structure: Dict[str, Any]
    backend_structure: Dict[str, Any]
    database_schema: Dict[str, Any]
    deployment: Dict[str, Any]
    rationale: str
    project_structure: Dict[str, Any]  # Expected: {"backend": ["main.py", ...], "frontend": ["src/App.js", ...]}
