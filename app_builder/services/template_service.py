import os
import json
from typing import Dict, Any, Optional

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")

TEMPLATE_MAPPING = {
    "todo-list": {
        "keywords": ["todo", "to-do", "task list", "checklist", "todolist"],
        "file_path": os.path.join(TEMPLATES_DIR, "todo-list", "todolist.json")
    },
    "vacation-budget-planner": {
        "keywords": ["vacation", "holiday", "budget planner", "trip planner"],
        "file_path": os.path.join(TEMPLATES_DIR, "vacation-budget-planner", "vacation-budget-planner.json")
    }
}

def detect_template(requirement: str) -> Optional[str]:
    """
    Detects which template to use based on the requirement string.
    Returns the template name if a match is found, otherwise None.
    """
    req_lower = requirement.lower()
    for template_name, info in TEMPLATE_MAPPING.items():
        if any(kw in req_lower for kw in info["keywords"]):
            return template_name
    return None

def load_template(template_name: str) -> Optional[Dict[str, Any]]:
    """
    Loads the JSON content of the specified template.
    """
    if template_name not in TEMPLATE_MAPPING:
        return None
    
    file_path = TEMPLATE_MAPPING[template_name]["file_path"]
    if not os.path.exists(file_path):
        return None
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading template {template_name}: {e}")
        return None
