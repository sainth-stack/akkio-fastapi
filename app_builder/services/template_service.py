"""
Template Service - Two templates: ideas-generator (LLM) and todo-list (normal).
Templates live in app_builder/templates.
"""
import logging
import os

logger = logging.getLogger("app_builder")
import json
from typing import Dict, Any, Optional

TEMPLATES_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")

# Only two templates: LLM (ideas-generator) and normal (todo-list)
TEMPLATE_MAPPING = {
    "ideas-generator": {
        "keywords": [
            "ideas generator", "idea generator", "generate ideas", "brainstorm", "creative ideas", "startup ideas",
            "linkedin post", "linkedin post generator", "linkedin-post-generator", "linkedin-post", "generate linkedin post", "linkedin content", "social media post",
            "travel planner", "travel plan", "trip planner", "vacation", "itinerary", "travel suggestion",
            "translate", "translation", "language translator", "translator",
            "genai", "gen ai", "ai generator", "ai post", "content generator",
        ],
        "template_path": "ideas-generator",
        "config": "ideas-generator.json",
    },
    "todo-list": {
        "keywords": ["todo", "to-do", "todo list", "task list", "checklist", "todolist", "tasks", "list application"],
        "template_path": "todo-list",
        "config": "todolist.json",
    },
}


def detect_template(requirement: str) -> Optional[str]:
    """Detects which template to use based on the requirement string."""
    req_lower = (requirement or "").lower()
    for template_name, info in TEMPLATE_MAPPING.items():
        if any(kw in req_lower for kw in info["keywords"]):
            logger.info("[template] detect_template match | requirement=%r -> %s", requirement[:80], template_name)
            return template_name
    logger.info("[template] detect_template no match | requirement=%r", requirement[:80] if requirement else "")
    return None


def _read_file(base_path: str, rel_path: str) -> Optional[str]:
    """Read a file from template directory."""
    full = os.path.join(base_path, rel_path)
    if not os.path.isfile(full):
        return None
    try:
        with open(full, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _collect_template_files(template_dir: str) -> Dict[str, str]:
    """Collect all source files from a template directory."""
    base = os.path.join(TEMPLATES_BASE, template_dir)
    if not os.path.isdir(base):
        return {}

    files = {}
    exclude_dirs = ("__pycache__", "node_modules", ".git", ".venv", "venv", "dist", "build")
    for root, dirs, filenames in os.walk(base):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        rel_root = os.path.relpath(root, base)
        for fn in filenames:
            if fn.startswith(".") or fn.endswith(".pyc"):
                continue
            if fn.endswith(".json") and fn in ("translator.json", "todolist.json", "travel-planner.json", "ideas-generator.json"):
                continue
            rel_path = os.path.join(rel_root, fn) if rel_root != "." else fn
            rel_path = rel_path.replace("\\", "/")
            content = _read_file(base, rel_path)
            if content is not None:
                if rel_path.startswith("backend/") or rel_path.startswith("frontend/"):
                    files[rel_path] = content
                elif rel_path.startswith("backend"):
                    files["backend/" + rel_path[len("backend"):].lstrip("/")] = content
                elif rel_path.startswith("frontend"):
                    files["frontend/" + rel_path[len("frontend"):].lstrip("/")] = content
    return files


def load_template(template_name: str) -> Optional[Dict[str, Any]]:
    """Loads the JSON config for a template."""
    if template_name not in TEMPLATE_MAPPING:
        return None
    info = TEMPLATE_MAPPING[template_name]
    template_path = info.get("template_path")
    config_name = info.get("config")
    if not template_path or not config_name:
        return None
    content = _read_file(os.path.join(TEMPLATES_BASE, template_path), config_name)
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error loading template {template_name}: {e}")
        return None


def get_template_code_files(template_name: str) -> Optional[Dict[str, str]]:
    """
    Returns actual code files from app_builder/templates for code generation.
    Used when a template match is detected - code creation uses these files instead of LLM.
    """
    info = TEMPLATE_MAPPING.get(template_name)
    if not info or not info.get("template_path"):
        logger.warning("[template] get_template_code_files | unknown template=%s", template_name)
        return None
    template_dir = info["template_path"]
    raw = _collect_template_files(template_dir)
    if not raw:
        logger.warning("[template] get_template_code_files | no files found for %s (path=%s)", template_name, template_dir)
        return None
    out = _prepare_template_for_generation(raw)
    logger.info("[template] get_template_code_files | template=%s | files=%d", template_name, len(out))
    return out


def _prepare_template_for_generation(raw: Dict[str, str]) -> Dict[str, str]:
    """Ensure standard frontend scaffold, index.js, index.html, package.json."""
    files = dict(raw)
    if "frontend/App.js" in files and "frontend/src/App.js" not in files:
        files["frontend/src/App.js"] = files["frontend/App.js"]
        del files["frontend/App.js"]
    for old, new in [
        ("frontend/App.js", "frontend/src/App.js"),
        ("frontend/styles.css", "frontend/src/styles.css"),
    ]:
        if old in files and new not in files:
            files[new] = files[old]
            del files[old]
    if "frontend/src/index.js" not in files:
        files["frontend/src/index.js"] = """import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><App /></React.StrictMode>);
"""
    if "frontend/public/index.html" not in files:
        files["frontend/public/index.html"] = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
    <script src="https://cdn.tailwindcss.com"></script>
    <title>App</title>
  </head>
  <body><div id="root"></div></body>
</html>
"""
    if "frontend/package.json" not in files:
        files["frontend/package.json"] = json.dumps({
            "name": "frontend",
            "version": "0.1.0",
            "private": True,
            "dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0", "react-scripts": "5.0.1"},
            "scripts": {
                "start": "HOST=0.0.0.0 DANGEROUSLY_DISABLE_HOST_CHECK=true PORT=5002 NODE_OPTIONS=--openssl-legacy-provider react-scripts start",
                "build": "NODE_OPTIONS=--openssl-legacy-provider react-scripts build",
            },
            "engines": {"node": ">=20"},
        }, indent=2)
    return files
