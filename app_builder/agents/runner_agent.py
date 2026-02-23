import os
import re
import json
import asyncio
from typing import Dict, List, Any, Optional
from llm_helper import ainvoke_llm_for_user

async def auto_fix_error(project_name: str, project_root: str, error_logs: str, component: str) -> bool:
    """
    Use LLM to analyze error logs and fix the code.
    component: 'backend' or 'frontend'
    """
    print(f"Auto-fixing {component} error for project {project_name}...")
    
    # 1. Get project context (list of files)
    all_files = []
    for root, dirs, filenames in os.walk(project_root):
        if any(ex in root for ex in ["node_modules", "__pycache__", ".git", ".next", "venv", "dist", "build"]):
            continue
        for f in filenames:
            rel_path = os.path.relpath(os.path.join(root, f), project_root)
            all_files.append(rel_path)
    
    # 2. Prepare prompt for LLM
    prompt = f"""
    You are an expert developer assistant. An application project "{project_name}" failed to start.
    
    Component: {component}
    Error Logs:
    {error_logs}
    
    Project Files:
    {", ".join(all_files)}
    
    CRITICAL CONSTRAINTS (Python 3.13):
    - The app runs on Python 3.13. NEVER pin or downgrade package versions.
    - NEVER add version numbers to requirements.txt (no ==, >=, ~=).
    - NEVER use Motor (incompatible with Python 3.13). Use PyMongo (sync) for MongoDB.
    - Use Pydantic v2 patterns: model_config = ConfigDict(...), .model_dump(), from_attributes=True.
    - NEVER use Pydantic v1 patterns: class Config, orm_mode, .dict().
    
    Your task:
    1. Analyze the error logs to identify the root cause.
    2. Suggest the exact code changes needed to fix the error.
    3. You must respond ONLY with a JSON object in the following format:
    {{
        "analysis": "Brief analysis of the error",
        "fixes": [
            {{
                "path": "path/to/file/relative/to/project/root",
                "explanation": "Why this fix is needed",
                "new_content": "Full content of the file after fix"
            }}
        ]
    }}
    
    Keep the changes minimal but effective. Ensure the app can run after these changes.
    """
    
    try:
        response_text = await ainvoke_llm_for_user(None, prompt, temperature=0.2)
        # Extract JSON from response if it's wrapped in markdown
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            response_json = json.loads(json_match.group(1))
        else:
            # Try parsing the whole thing as JSON
            response_json = json.loads(response_text.strip())
        
        fixes = response_json.get("fixes", [])
        if not fixes:
            print("No fixes suggested by LLM.")
            return False
        
        for fix in fixes:
            file_path = os.path.join(project_root, fix["path"])
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fix["new_content"])
            print(f"Applied fix to {fix['path']}: {fix['explanation']}")
            
        return True
        
    except Exception as e:
        print(f"Error during auto-fix: {e}")
        import traceback
        traceback.print_exc()
        return False
