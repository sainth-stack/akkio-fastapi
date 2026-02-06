"""
Dynamic Code Generator Agent - Generates code based on PRD and implementation plan using LLM
"""
from typing import Dict, Any, AsyncGenerator
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage

VITE_REACT_PLUGIN = "@vitejs/plugin-react"

# Minimal CRA package.json with pinned versions so generated app runs without version issues
_CRA_PACKAGE_JSON = {
    "name": "frontend",
    "version": "1.0.0",
    "private": True,
    "dependencies": {
        "react": "18.2.0",
        "react-dom": "18.2.0",
        "react-scripts": "5.0.1",
        "web-vitals": "^2.1.4",
        "tailwindcss": "^3.4.1",
        "postcss": "^8.4.35",
        "autoprefixer": "^10.4.17",
        "framer-motion": "^10.16.4",
        "lucide-react": "^0.263.1",
        "axios": "^1.5.0",
        "clsx": "^2.0.0",
        "tailwind-merge": "^2.2.1"
    },
    "scripts": {
        "start": "react-scripts start",
        "build": "react-scripts build",
    },
    "eslintConfig": {"extends": ["react-app"]},
    "browserslist": {
        "production": [">0.2%", "not dead", "not op_mini all"],
        "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"],
    },
}


def _normalize_frontend_package_json_to_cra(files: Dict[str, str]) -> None:
    """
    If generated frontend uses Vite or problematic deps (@dnd-kit, etc.), replace
    with minimal Create React App (react-scripts) package.json so the app runs.
    Remove vite.config and ensure CRA entry files (public/index.html, src/index.js) exist.
    """
    pkg_path = "frontend/package.json"
    if pkg_path not in files:
        return
    try:
        pkg = json.loads(files[pkg_path])
        scripts = pkg.get("scripts") or {}
        deps = pkg.get("dependencies") or {}
        dev_deps = pkg.get("devDependencies") or {}
        has_vite = "vite" in str(scripts).lower() or "vite" in str(dev_deps).lower()
        has_dnd = any(k.startswith("@dnd-kit") for k in list(deps.keys()) + list(dev_deps.keys()))
        if not has_vite and not has_dnd:
            return
        cra = dict(_CRA_PACKAGE_JSON)
        cra["name"] = pkg.get("name", "frontend")
        files[pkg_path] = json.dumps(cra, indent=2)
        for vpath in ("frontend/vite.config.js", "frontend/vite.config.ts"):
            if vpath in files:
                del files[vpath]
        # Ensure CRA entry points exist (react-scripts expects public/index.html, src/index.js)
        if "frontend/public/index.html" not in files and "frontend/index.html" in files:
            files["frontend/public/index.html"] = files["frontend/index.html"]
        if "frontend/public/index.html" not in files:
            files["frontend/public/index.html"] = (
                "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
                "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>"
                "<title>React App</title></head><body><noscript>You need to enable JavaScript.</noscript>"
                "<div id=\"root\"></div></body></html>"
            )
        if "frontend/src/index.js" not in files and "frontend/src/index.jsx" not in files:
            files["frontend/src/index.js"] = (
                "import React from 'react';\nimport ReactDOM from 'react-dom/client';\n"
                "import App from './App';\nconst root = ReactDOM.createRoot(document.getElementById('root'));\n"
                "root.render(<React.StrictMode><App /></React.StrictMode>);"
            )
    except (json.JSONDecodeError, TypeError, KeyError):
        pass


# Pinned versions for backend to avoid install/version issues (simple, stable)
_BACKEND_PINNED = {
    "fastapi": "0.109.2",
    "uvicorn": "0.27.1",
    "sqlalchemy": "2.0.25",
    "psycopg2-binary": "2.9.9",
    "pydantic": "1.10.13",
}


def _pin_backend_requirements(files: Dict[str, str]) -> None:
    """
    Normalize backend/requirements.txt to use pinned versions for common packages
    so installs run without version conflicts. Prefer simple, stable versions.
    """
    path = "backend/requirements.txt"
    if path not in files:
        return
    lines = []
    seen = set()
    for raw in files[path].strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            lines.append(line)
            continue
        # Parse "package" or "package==x" or "package>=x" or "package[extra]"
        pkg = line.split("==")[0].split(">=")[0].split("[")[0].strip().lower()
        if pkg in _BACKEND_PINNED:
            if pkg not in seen:
                seen.add(pkg)
                lines.append(f"{pkg}=={_BACKEND_PINNED[pkg]}")
            # skip duplicate or alternate spec for same package
        else:
            lines.append(line)
    files[path] = "\n".join(lines) + "\n"


def _fix_backend_python_relative_imports(files: Dict[str, str]) -> None:
    """
    Backend is run as 'uvicorn main:app' from the backend directory, so all .py
    files are loaded as top-level modules, not as part of a package. Relative
    imports (from . import ...) then fail. Convert them to absolute imports
    in every backend/*.py file so the app starts.
    """
    for path in list(files.keys()):
        if not (path.startswith("backend/") and path.endswith(".py")):
            continue
        content = files[path]
        if "from . import " not in content and "from ." not in content:
            continue
        content = re.sub(r"\bfrom\s+\.\s+import\s+", "import ", content)
        content = re.sub(r"\bfrom\s+\.(\w+)\s+import\s+", r"from \1 import ", content)
        files[path] = content


def _ensure_vite_react_plugin_in_package_json(files: Dict[str, str]) -> None:
    """
    If the project has a Vite frontend (vite.config.js/ts), ensure frontend/package.json
    includes @vitejs/plugin-react in devDependencies. The LLM sometimes omits it even when
    vite.config uses it, which causes 'Cannot find module @vitejs/plugin-react' at runtime.
    Mutates files in place so generated output is always runnable without manual installs.
    """
    has_vite_config = (
        "frontend/vite.config.js" in files or "frontend/vite.config.ts" in files
    )
    pkg_path = "frontend/package.json"
    if not has_vite_config or pkg_path not in files:
        return
    try:
        pkg = json.loads(files[pkg_path])
        dev_deps = pkg.get("devDependencies") or {}
        if dev_deps.get(VITE_REACT_PLUGIN):
            return
        dev_deps[VITE_REACT_PLUGIN] = "^4.2.1"
        pkg["devDependencies"] = dev_deps
        files[pkg_path] = json.dumps(pkg, indent=2)
    except (json.JSONDecodeError, TypeError):
        pass

async def generate_code_from_plan(
    requirement: str,
    prd: str,
    plan: list,
    architecture: Dict[str, Any],
    llm,
    uiux: str = ""
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generates actual code based on PRD and implementation plan using LLM.
    
    Args:
        requirement: Original user requirement
        prd: Complete PRD document
        plan: Implementation plan with steps
        architecture: Architecture decisions
        llm: Initialized LLM instance
        
    Yields:
        Dictionary events with file generation progress
    """
    
    # Extract key information from plan
    plan_summary = "\n".join([
        f"{i+1}. {step.get('title', step) if isinstance(step, dict) else step}"
        for i, step in enumerate(plan[:20])  # Increased context
    ])
    
    # Extract database schema
    db_schema = architecture.get('database_schema', {})
    tables = db_schema.get('tables', [])
    
    # Parse architecture details
    project_structure = architecture.get("project_structure", {})
    raw_backend_files = project_structure.get("backend", [])
    raw_frontend_files = project_structure.get("frontend", [])
    
    # Ensure all files have the correct directory prefix
    backend_files = []
    for f in raw_backend_files:
        if f.startswith("backend/"):
            backend_files.append(f)
        else:
            backend_files.append(f"backend/{f}")
            
    frontend_files = []
    for f in raw_frontend_files:
        if f.startswith("frontend/"):
            frontend_files.append(f)
        else:
            frontend_files.append(f"frontend/{f}")
    
    backend_framework = architecture.get('backend_structure', {}).get('framework', 'Unknown')
    frontend_framework = architecture.get('frontend_structure', {}).get('framework', 'Unknown')
    
    # Format architecture summary for context
    arch_summary = json.dumps(architecture, indent=2)
    
    # Construct the system prompt
    system_prompt = f"""You are an expert full-stack developer. You must generate a complete, working application based on the requirements and architecture.

**PRODUCTION READY CODE – GENERATE HIGH-QUALITY, PREMIUM APPLICATIONS:**
1. **Visual Excellence (CRITICAL)**:
   - **Whitespace**: Use "Airy" designs. Use `p-6`, `p-8`, `gap-6` generously. Avoid cramped UIs.
   - **Shadows & Depth**: Use `shadow-sm` or `shadow-md` with `border border-gray-100` for cards. Avoid flat, baseless designs.
   - **Typography**: Use `text-gray-900` for headings and `text-gray-500` for supporting text. NEVER use pure black (#000000).
   - **Rounding**: Use `rounded-lg` or `rounded-xl` for modern feel.
2. **Interactive & Alive**:
   - Add `:hover` states to ALL interactive elements (buttons, rows, cards).
   - Use `transition-all duration-200` for smooth interactions.
3. **Robust Layouts**:
   - **Sidebar**: If implementing a sidebar, make it full height `h-screen`, fixed or sticky.
   - **Navbar**: Make it `sticky top-0 z-50` with a subtle blur `backdrop-blur-md bg-white/80` or solid white with border.
4. **No Placeholders**: **NEVER** use comments like `# implementation here`. Write the FULL code.
5. **Design Fidelity**: Follow the UI/UX colors and fonts EXACTLY.

**STRICT FILE GENERATION RULES:**
6. Generate code for every file listed in the project structure.
7. **Dependencies**:
   - For React: Use `react-scripts`. NO VITE.
   - For Backend: Use `fastapi`, `uvicorn`, `sqlalchemy`.
   - **CRITICAL**: You must ensure that every import you use in the code is also added to `package.json` or `requirements.txt`.
   - **Tailwind**: You MUST include `tailwindcss`, `postcss`, `autoprefixer` in `package.json`.
   - **Icons**: Use `lucide-react` for all icons.
8. **Files**:
   - `frontend/package.json`: MUST include `framer-motion`, `lucide-react`, `clsx`, `tailwind-merge`.
   - `backend/requirements.txt`: MUST include all used python packages.
   - `backend/main.py`: Use absolute imports.
9. **Structure**:
   - Follow the provided architecture structure exactly.

**UI/UX Design Specification:**
{uiux if uiux else "Standard modern UI/UX design (CLEAN, MODERN, USER-FRIENDLY)."}

**CRITICAL STYLE INSTRUCTIONS:**
1. **TAILWIND CSS MANDATORY**:
   - You MUST generate a `tailwind.config.js` file with the exact theme extensions (colors, fonts).
   - You MUST ensure `src/index.css` (or `src/App.css`) includes the standard Tailwind directives.
   - **Use Tailwind Utility Classes**: `bg-indigo-600`, `text-slate-900`, `ring-1 ring-slate-900/5`.
   - **Avoid Custom CSS**: Use Tailwind for everything.

2. **UI/UX COMPLIANCE**:
   - **EXTRACT** the exact Hex Codes for Primary, Secondary, Background, and Text colors from the "UI/UX Design Specification".
   - **APPLY** these exact colors to your `tailwind.config.js` theme.
   - **Layout**: 
     - IF UI/UX SAYS "Sidebar + Top Nav" -> YOU MUST BUILD key layout components.
     - IF UI/UX SAYS "Dashboard" -> Build a proper dashboard grid.


3. **ROBUSTNESS**:
   - Ensure the app is "npm install" ready. All deps used in code must be in `package.json`.


**Target Architecture:**
- Backend Framework: {backend_framework}
- Frontend Framework: {frontend_framework}
- Database: {architecture.get('backend_structure', {}).get('database', 'Unknown')}

**Expected Backend Files:**
{json.dumps(backend_files, indent=2)}

**Expected Frontend Files:**
{json.dumps(frontend_files, indent=2)}

**Output Format:**
- Use the `FILE: <path>` format for every file.
- Followed by the code block.

**Example:**
FILE: backend/main.py
```python
from fastapi import FastAPI
...
```

**Architecture Context:**
{arch_summary}

**Dependencies:**
- Use pinned versions to avoid version issues: backend (e.g. fastapi==0.109.2, uvicorn==0.27.1, sqlalchemy==2.0.25, pydantic==1.10.13).
- **Frontend Stable Versions (USE THESE):**
  - `lucide-react`: `^0.263.1` (NOT 0.1.0)
  - `framer-motion`: `^10.16.4`
  - `react-router-dom`: `^6.16.0`
  - `axios`: `^1.5.0`
  - `clsx`: `^2.0.0`
  - `tailwind-merge`: `^1.14.0`
- In README.md installation steps use: `pip install -r requirements.txt --force-reinstall` and `npm install --force` so installs succeed reliably.
"""

    user_prompt = f"""Generate complete application code for this SPECIFIC requirement:

## REQUIREMENT:
{requirement}

## PRD (Key Points):
{prd[:3000]}

## Implementation Steps:
{plan_summary}

## Architecture & Database Schema:
{arch_summary}

DATABASE TABLES TO IMPLEMENT:
{json.dumps(tables, indent=2) if tables else 'Use entities from requirement'}

IMPORTANT:
- Generate code for the ACTUAL requirement.
- STRICTLY follow the file paths in "project_structure" – generate only those files; do not add extra components or files.
- If the architecture says "NestJS", generate "NestJS" code. If it says "FastAPI", generate "FastAPI" code.
- Ensure all imports match the file structure.
- Prefer a single App file with all UI when the requirement is simple; create separate components only when required.

Start generating strictly using the "FILE: <path>" format.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    yield {
        "event": "generation_start",
        "message": "Starting code generation based on PRD and plan..."
    }
    
    # Stream the code generation
    full_response = ""
    file_starts = []  # List of (start_index, file_path)
    files_yielded = 0
    files_generated = {}  # Initialize to fix NameError

    
    chunk_count = 0
    
    async for chunk in llm.astream(messages):
        if hasattr(chunk, 'content'):
            content = chunk.content
            if not content:
                continue
                
            # append content
            prev_len = len(full_response)
            full_response += content
            chunk_count += 1
            
            # Progress update removed to prevent continuous messaging

            # Search for "FILE:" markers in the newly added content 
            # (including a bit of overlap context to catch split markers)
            search_start = max(0, prev_len - 10)
            
            # Simple check: Does the new content contain "FILE:"? 
            # We scan the full_response from search_start to find new markers
            import re
            # Regex to find "FILE: <path>" at start of line or string
            # We use a simple find loop for robustness
            
            lines = full_response[search_start:].split('\n')
            
            # This is slightly inefficient but safe: check if we found a new FILE: line
            # We track known file starts to avoid duplicates
            
            # Better approach: Scan full_response for all markers, compare with known ones
            # For performance, we can just scan the tail, but "FILE:" is rare enough.
            
            # Let's iterate lines in full text for simplicity and correctness over micro-optimization
            all_lines = full_response.split('\n')
            current_file_starts = []
            
            for i, line in enumerate(all_lines):
                if line.strip().startswith("FILE:"):
                    path = line.replace("FILE:", "").strip()
                    current_file_starts.append((i, path))
            
            # Check if we have a NEW completed file
            # A file is completed if we have a marker AFTER it
            if len(current_file_starts) > files_yielded + 1:
                # We have at least one completed file that hasn't been yielded
                # The file at files_yielded is now complete because files_yielded+1 exists
                
                # Extract content for the file at 'files_yielded'
                start_line_idx, file_path_to_yield = current_file_starts[files_yielded]
                end_line_idx, _ = current_file_starts[files_yielded + 1]
                
                # Content is lines between start and end
                file_lines = all_lines[start_line_idx+1 : end_line_idx]
                file_content = "\n".join(file_lines).strip()
                
                # Clean up markdown code blocks
                if "```" in file_content:
                    parts = file_content.split("```")
                    if len(parts) >= 2:
                        code = parts[1]
                        if code.strip() and "\n" in code: # typical ```python\n code...
                             code = "\n".join(code.split("\n")[1:])
                        file_content = code.strip()
                    else:
                        file_content = file_content.replace("```", "").strip()
                
                files_yielded += 1

                # Skip yielding directories
                if file_path_to_yield.endswith("/") or file_path_to_yield.endswith("\\"):
                    continue

                files_generated[file_path_to_yield] = file_content
                yield {
                    "event": "file_generated",
                    "file": file_path_to_yield,
                    "content": file_content,
                    "message": f"Generated {file_path_to_yield}"
                }
                 

    # Handled yielded files above loop. 
    # Now verify remaining file (the last one)
    all_lines = full_response.split('\n')
    current_file_starts = []
    for i, line in enumerate(all_lines):
        if line.strip().startswith("FILE:"):
            path = line.replace("FILE:", "").strip()
            current_file_starts.append((i, path))
            
    if len(current_file_starts) > files_yielded:
        # Yield the last file
        start_line_idx, file_path_to_yield = current_file_starts[files_yielded]
        # Content is distinct from start to end of string
        file_lines = all_lines[start_line_idx+1:]
        file_content = "\n".join(file_lines).strip()
        
        # Clean up markdown
        if "```" in file_content:
            parts = file_content.split("```")
            if len(parts) >= 2:
                code = parts[1]
                if code.strip() and "\n" in code:
                     code = "\n".join(code.split("\n")[1:])
                file_content = code.strip()
            # Handle case where closing ``` is missing (end of stream)
            elif len(parts) == 1 and parts[0].strip():
                 # Maybe header ``` is there but no footer
                 pass 
        
        file_content = file_content.replace("```", "").strip() # fallback cleanup
        
        files_generated[file_path_to_yield] = file_content
        yield {
            "event": "file_generated",
            "file": file_path_to_yield,
            "content": file_content,
            "message": f"Generated {file_path_to_yield}"
        }

    # Prefer CRA (react-scripts) over Vite: if generated frontend has Vite or @dnd-kit, replace with minimal runnable CRA
    _normalize_frontend_package_json_to_cra(files_generated)
    # Ensure Vite + React projects have @vitejs/plugin-react in package.json (only if we didn't replace with CRA)
    _ensure_vite_react_plugin_in_package_json(files_generated)
    # Pin backend requirements to stable versions so pip install runs without version issues
    _pin_backend_requirements(files_generated)
    # Fix all backend .py relative imports so uvicorn main:app works (no parent package).
    _fix_backend_python_relative_imports(files_generated)

    yield {
        "event": "generation_complete",
        "data": files_generated,
        "message": f"Generated {len(files_generated)} files"
    }
