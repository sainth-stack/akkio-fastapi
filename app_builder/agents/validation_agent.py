import logging
import re
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger("app_builder")

from app_builder.agents.dynamic_code_generator import (
    _fix_frontend_map_safety,
    _fix_frontend_backend_url_undefined,
    _fix_backend_routes_import,
)


# Enhanced styles applied when using templates - improves spacing, alignment, format
POLISHED_STYLES_CSS = """/* Polished template styles - improved spacing, alignment, and format */
:root {
  --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
  --color-bg: #f8fafc;
  --color-surface: #ffffff;
  --color-primary: #6366f1;
  --color-primary-hover: #4f46e5;
  --color-text: #1e293b;
  --color-text-muted: #64748b;
  --color-border: #e2e8f0;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --spacing-xs: 0.25rem;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;
  --spacing-xl: 2rem;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: var(--font-sans);
  background: var(--color-bg);
  color: var(--color-text);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.app {
  padding: var(--spacing-xl) var(--spacing-lg);
  max-width: 720px;
  margin: 0 auto;
  min-height: 100vh;
}

.app-header {
  margin-bottom: var(--spacing-xl);
  text-align: center;
}
.app-title {
  font-size: 2rem;
  font-weight: 700;
  margin: 0 0 var(--spacing-sm) 0;
  letter-spacing: -0.025em;
}
.app-subtitle {
  color: var(--color-text-muted);
  font-size: 1rem;
  margin: 0;
}

.section { margin-bottom: var(--spacing-xl); }
.section-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: var(--spacing-md);
  padding-bottom: var(--spacing-sm);
  border-bottom: 2px solid var(--color-border);
}

.card {
  background: var(--color-surface);
  padding: var(--spacing-lg);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  border: 1px solid var(--color-border);
  margin-bottom: var(--spacing-lg);
}
.todo-card {
  border: 1px solid rgba(255, 255, 255, 0.5);
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(12px);
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-md);
}
@media (max-width: 500px) {
  .form-grid { grid-template-columns: 1fr; }
}
.input-group { margin-bottom: var(--spacing-md); }
.input-label {
  display: block;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--color-text-muted);
  margin-bottom: var(--spacing-xs);
}
.input {
  width: 100%;
  padding: 0.625rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-family: inherit;
  font-size: 1rem;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.input:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.625rem 1.25rem;
  border-radius: var(--radius-md);
  font-weight: 600;
  cursor: pointer;
  border: none;
  font-size: 0.875rem;
  transition: all 0.2s;
}
.btn-primary {
  background: var(--color-primary);
  color: white;
}
.btn-primary:hover {
  background: var(--color-primary-hover);
  transform: translateY(-1px);
}
.btn-danger {
  background: #fee2e2;
  color: #dc2626;
}
.btn-danger:hover { background: #fecaca; }

.filter-tabs {
  display: flex;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-md);
  flex-wrap: wrap;
}
.filter-tab {
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
  background: white;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}
.filter-tab:hover,
.filter-tab.active {
  background: var(--color-primary);
  color: white;
  border-color: var(--color-primary);
}

.list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}
.list-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--spacing-md) var(--spacing-lg);
  background: white;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  transition: all 0.2s;
}
.list-item:hover {
  transform: translateX(4px);
  border-color: var(--color-primary);
  box-shadow: var(--shadow-md);
}

.todo-item { gap: var(--spacing-md); }
.todo-item-left {
  display: flex;
  align-items: flex-start;
  gap: var(--spacing-md);
  flex: 1;
}
.todo-checkbox {
  width: 1.25rem;
  height: 1.25rem;
  margin-top: 0.2rem;
  cursor: pointer;
  accent-color: var(--color-primary);
}
.todo-text { font-weight: 500; }
.todo-text.completed {
  text-decoration: line-through;
  opacity: 0.7;
}
.todo-desc {
  font-size: 0.875rem;
  color: var(--color-text-muted);
  margin-top: var(--spacing-xs);
}
.todo-item-completed { opacity: 0.85; }
.todo-badge {
  font-size: 0.7rem;
  padding: 0.2rem 0.5rem;
  border-radius: 9999px;
  font-weight: 600;
  text-transform: uppercase;
}
.todo-badge-high { background: #fecaca; color: #b91c1c; }
.todo-badge-medium { background: #fef3c7; color: #b45309; }
.todo-badge-low { background: #d1fae5; color: #047857; }
"""


def polish_template_output(files: Dict[str, str], template_name: str) -> Dict[str, str]:
    """
    Improves template output: styles, format, alignment.
    - Applies enhanced CSS for todo-list/travel-planner (light theme with consistent spacing).
    - For language-translator/ideas-generator keeps their styles but appends alignment utilities.
    - Normalizes code formatting (trim, collapse blanks).
    """
    result = dict(files)
    # 1. Enhance styles.css - full polished version for todo/travel; append utilities for others
    for path in ["frontend/src/styles.css", "frontend/styles.css"]:
        if path not in result:
            continue
        existing = result[path]
        if template_name in ("todo-list", "travel-planner", "vacation-budget-planner"):
            result[path] = POLISHED_STYLES_CSS + _travel_planner_extras() if "trip-card" in existing else POLISHED_STYLES_CSS
        else:
            # Append alignment/format utilities without replacing template's design
            if ".text-center" not in existing and "align-items: center" not in existing:
                result[path] = existing.rstrip() + "\n\n/* Alignment utilities */\n.app { display: flex; flex-direction: column; align-items: center; }\n.app-header { width: 100%; }\n"
        break
    # 2. Normalize formatting in frontend JS/JSX
    for path, content in list(result.items()):
        if path.startswith("frontend/") and path.endswith((".js", ".jsx", ".ts", ".tsx")):
            result[path] = _normalize_js_format(content)
    return result


def _travel_planner_extras() -> str:
    """Extra CSS for travel-planner template (trip-card, itinerary-item)."""
    return """
.travel-app { max-width: 800px; margin: 0 auto; padding: var(--spacing-xl) var(--spacing-lg); }
.trip-cards { display: flex; flex-wrap: wrap; gap: var(--spacing-md); margin-top: var(--spacing-md); }
.trip-card { padding: var(--spacing-md); border: 1px solid var(--color-border); border-radius: var(--radius-md); cursor: pointer; transition: all 0.2s; }
.trip-card:hover, .trip-card.selected { border-color: var(--color-primary); box-shadow: var(--shadow-md); }
.trip-card-title { margin: 0 0 var(--spacing-xs) 0; font-size: 1rem; font-weight: 600; }
.trip-card-dest { margin: 0; font-size: 0.875rem; color: var(--color-text-muted); }
.trip-card-dates { margin: var(--spacing-xs) 0; font-size: 0.8rem; color: var(--color-text-muted); }
.trip-card-budget { margin: var(--spacing-xs) 0 0 0; font-weight: 600; color: var(--color-primary); }
.itinerary-list { display: flex; flex-direction: column; gap: var(--spacing-sm); margin-top: var(--spacing-md); }
.itinerary-item { display: flex; align-items: flex-start; justify-content: space-between; padding: var(--spacing-md); background: white; border: 1px solid var(--color-border); border-radius: var(--radius-md); }
.itinerary-day { font-size: 0.75rem; font-weight: 600; color: var(--color-primary); }
.itinerary-time { font-size: 0.8rem; color: var(--color-text-muted); margin-right: var(--spacing-sm); }
.itinerary-title { display: block; margin: 0 0 var(--spacing-xs) 0; }
.itinerary-desc { margin: 0; font-size: 0.875rem; color: var(--color-text-muted); }
.btn-sm { padding: 0.375rem 0.75rem; font-size: 0.8rem; }
.textarea { min-height: 80px; resize: vertical; }
"""


def _normalize_js_format(content: str) -> str:
    """Normalize format: trim, collapse excessive blanks, ensure trailing newline."""
    lines = content.split("\n")
    out = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank:
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(line.rstrip())
            prev_blank = False
    result = "\n".join(out).strip()
    return result + "\n" if result else ""


def validate_and_fix_code(
    files: Dict[str, str],
    architecture: Dict[str, Any],
    template_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Validates and attempts to fix the generated code.
    For ideas-generator template, verifies generate-ideas API contract (topic, count, gen_type -> content[]).
    """
    
    # 0. Fix cross-file imports FIRST (most common LLM error)
    _fix_backend_cross_file_imports(files)
    
    # 1. Backend Validation
    _validate_backend_dependencies(files)
    
    # 2. Frontend Validation
    _validate_frontend_dependencies(files)
    
    # 3. Frontend .map() safety - prevent "X.map is not a function"
    _fix_frontend_map_safety(files)
    # 4. Frontend API URL - prevent 404 on /undefined/tasks/
    _fix_frontend_backend_url_undefined(files)
    
    # 5. Backend routes import - prevent "cannot import todo_router from routes"
    _fix_backend_routes_import(files)
    
    # 6. Frontend fetch() safety - wrap in try/catch to prevent crash when backend is down
    _fix_frontend_fetch_safety(files)
    
    # 7. Fix Zustand import (default -> named export for v4+)
    _fix_zustand_import(files)
    
    # 8. Custom endpoints: validate frontend matches API contract (any template with custom endpoints)
    if template_name:
        custom = {"generate-ideas": "llm_content", "generate": "llm_content"} if template_name == "ideas-generator" else {}
        if template_name == "language-translator":
            custom = {"translate": "llm_translate"}
        if custom:
            _validate_custom_endpoint_api(files, template_name, custom)
    
    return files


def _validate_ideas_generator_api(files: Dict[str, str]) -> None:
    """
    Validate ideas-generator frontend matches generate-ideas API contract:
    - Request: { topic, count?, gen_type? }
    - Response: { topic, gen_type, content: string[], raw_text? }
    - Frontend must: POST to .../generate-ideas, send body, use data.content
    """
    for path in ["frontend/src/App.js", "frontend/App.js"]:
        if path not in files:
            continue
        content = files[path]
        issues = []
        if "/generate-ideas" not in content:
            issues.append("missing /generate-ideas in fetch URL")
        if "topic" not in content or "gen_type" not in content:
            issues.append("request body should include topic and gen_type")
        if "data.content" not in content and "data?.content" not in content:
            issues.append("response must use data.content (array)")
        if issues:
            logger.warning("[validation] ideas-generator API contract issues in %s: %s", path, issues)
        break


def _fix_zustand_import(files: Dict[str, str]):
    """
    Fix Zustand import for v4+ compatibility.
    LLM often generates: import create from 'zustand'
    But Zustand v4+ requires: import { create } from 'zustand'
    """
    for path, content in list(files.items()):
        if not (path.startswith("frontend/") and path.endswith((".js", ".jsx", ".ts", ".tsx"))):
            continue
        # Fix default import to named import
        fixed = re.sub(
            r'''import\s+create\s+from\s+['"]zustand['"]''',
            "import { create } from 'zustand'",
            content
        )
        if fixed != content:
            files[path] = fixed


def _fix_frontend_fetch_safety(files: Dict[str, str]):
    """
    Wrap bare fetch() calls in frontend JS files with try/catch to prevent
    the React tree from crashing when the backend is unavailable.
    
    Catches patterns like:
        const response = await fetch('...');
        const data = await response.json();
    
    And wraps them in try/catch blocks.
    """
    for path, content in list(files.items()):
        if not (path.startswith("frontend/") and path.endswith((".js", ".jsx", ".ts", ".tsx"))):
            continue
        
        # Skip if no fetch calls
        if "fetch(" not in content:
            continue
        
        lines = content.split("\n")
        new_lines = []
        i = 0
        changed = False
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Detect: async function/arrow that contains bare fetch without try
            # Check if this line has "await fetch(" but is NOT inside a try block
            if "await fetch(" in stripped or "await response.json()" in stripped:
                # Look backward to see if we're already inside a try block
                inside_try = False
                brace_depth = 0
                for j in range(i - 1, max(i - 20, -1), -1):
                    prev = lines[j].strip()
                    brace_depth += prev.count('}') - prev.count('{')
                    if prev.startswith("try") and "{" in prev:
                        inside_try = True
                        break
                    if prev.startswith("} catch"):
                        break  # We're after a catch, not inside try
                
                if not inside_try and "await fetch(" in stripped:
                    # Find the indent
                    indent = line[:len(line) - len(line.lstrip())]
                    
                    # Collect consecutive lines that are part of this fetch block
                    fetch_block = [line]
                    k = i + 1
                    while k < len(lines):
                        next_stripped = lines[k].strip()
                        # Continue if it's part of fetch handling (response.json, set state, etc.)
                        if next_stripped and not next_stripped.startswith("//") and (
                            "await " in next_stripped or
                            "response" in next_stripped.lower() or
                            ".json()" in next_stripped or
                            "set" in next_stripped and "(" in next_stripped and "State" not in next_stripped or
                            next_stripped.startswith("const ") and "=" in next_stripped
                        ):
                            fetch_block.append(lines[k])
                            k += 1
                        else:
                            break
                    
                    # Wrap in try/catch
                    new_lines.append(f"{indent}try {{")
                    for fb_line in fetch_block:
                        new_lines.append(f"  {fb_line}")
                    new_lines.append(f"{indent}}} catch (err) {{")
                    new_lines.append(f"{indent}  console.warn('API call failed, using local data:', err.message);")
                    new_lines.append(f"{indent}}}")
                    i = k
                    changed = True
                    continue
            
            new_lines.append(line)
            i += 1
        
        if changed:
            files[path] = "\n".join(new_lines)


def _fix_backend_cross_file_imports(files: Dict[str, str]):
    """
    Fix cross-file import errors in backend Python files.
    
    The LLM commonly generates main.py with:
        from models import Todo, TodoCreate, TodoUpdate
    but TodoCreate/TodoUpdate are Pydantic schemas defined in schemas.py, not models.py.
    
    This function:
    1. Parses all `from X import A, B, C` in each backend/*.py
    2. Checks if A, B, C are actually defined in backend/X.py
    3. If not found, searches other backend files for the definition
    4. Rewrites the import to point at the correct file
    """
    backend_files = {
        path: content
        for path, content in files.items()
        if path.startswith("backend/") and path.endswith(".py")
    }
    if not backend_files:
        return

    # Build index: name -> backend module that defines it
    # Look for class Foo, def foo, Foo = ...
    defined_names: Dict[str, str] = {}  # name -> module (e.g. "schemas")
    for path, content in backend_files.items():
        module = path.replace("backend/", "").replace(".py", "")
        # class definitions
        for m in re.finditer(r'^class\s+(\w+)', content, re.MULTILINE):
            defined_names[m.group(1)] = module
        # function definitions
        for m in re.finditer(r'^def\s+(\w+)', content, re.MULTILINE):
            defined_names[m.group(1)] = module
        # top-level assignments (e.g. app = FastAPI(), engine = ...)
        for m in re.finditer(r'^(\w+)\s*=\s*', content, re.MULTILINE):
            name = m.group(1)
            if name not in defined_names:  # don't overwrite class/def
                defined_names[name] = module

    # Check each file's "from X import ..." statements
    for path, content in list(backend_files.items()):
        lines = content.split("\n")
        changed = False
        new_lines = []
        # Collect additional imports to add at the top
        extra_imports: Dict[str, List[str]] = {}  # module -> [names]

        for line in lines:
            match = re.match(r'^(\s*)from\s+(\w+)\s+import\s+(.+)$', line)
            if not match:
                new_lines.append(line)
                continue

            indent = match.group(1)
            source_module = match.group(2)
            imports_str = match.group(3)
            
            # Parse imported names (handle "A, B, C" and "A as X, B as Y")
            imported_names = [n.strip().split(" as ")[0].strip() 
                              for n in imports_str.split(",")]
            
            # Check if source module file exists
            source_path = f"backend/{source_module}.py"
            if source_path not in files:
                new_lines.append(line)
                continue
            
            source_content = files[source_path]
            
            # Check which names are missing from source
            found_in_source = []
            missing_from_source = []
            for name in imported_names:
                # Check if name is defined in source file (class, def, or assignment)
                pattern = rf'(?:^class\s+{re.escape(name)}\b|^def\s+{re.escape(name)}\b|^{re.escape(name)}\s*=)'
                if re.search(pattern, source_content, re.MULTILINE):
                    found_in_source.append(name)
                else:
                    missing_from_source.append(name)
            
            if not missing_from_source:
                # All imports are valid
                new_lines.append(line)
                continue
            
            changed = True
            
            # Keep valid imports from original source
            if found_in_source:
                new_lines.append(f"{indent}from {source_module} import {', '.join(found_in_source)}")
            
            # Find correct source for missing names
            for name in missing_from_source:
                correct_module = defined_names.get(name)
                if correct_module and correct_module != source_module:
                    extra_imports.setdefault(correct_module, []).append(name)
                else:
                    # Name not found anywhere - keep original import to avoid breaking more
                    if not found_in_source and not extra_imports:
                        new_lines.append(line)  # keep the line as-is
        
        if changed:
            # Add extra import lines
            import_lines = []
            for module, names in extra_imports.items():
                import_lines.append(f"from {module} import {', '.join(names)}")
            
            if import_lines:
                # Insert after existing imports at the top
                result_lines = []
                inserted = False
                for line in new_lines:
                    result_lines.append(line)
                    if not inserted and (line.startswith("import ") or line.startswith("from ")):
                        # Keep going until we find the last import
                        pass
                    elif not inserted and not line.startswith("import ") and not line.startswith("from ") and not line.strip() == "":
                        # Insert before first non-import, non-empty line
                        for imp_line in import_lines:
                            result_lines.insert(-1, imp_line)
                        inserted = True
                
                if not inserted:
                    # Fallback: prepend
                    result_lines = import_lines + result_lines
                
                files[path] = "\n".join(result_lines)
            else:
                files[path] = "\n".join(new_lines)


# Minimal frontend - no Tailwind, no heavy deps. Only add if explicitly needed.
KNOWN_STABLE_VERSIONS = {
    "react-router-dom": "^6.16.0",
    "axios": "^1.5.0",
}

# Packages to never add - use styles.css instead, keep minimal
FRONTEND_BLOCKLIST = {"tailwindcss", "postcss", "autoprefixer", "framer-motion", "lucide-react", "clsx", "tailwind-merge", "zustand"}

def _validate_backend_dependencies(files: Dict[str, str]):
    """
    Scan backend python files for imports and ensure they are in requirements.txt
    """
    req_path = "backend/requirements.txt"
    if req_path not in files:
        files[req_path] = "fastapi\nuvicorn\nsqlalchemy\npydantic\n"
    
    existing_reqs = set()
    req_content = files[req_path]
    for line in req_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Extract package name (remove version, extras)
        pkg = re.split(r'[=<>\[]', line)[0].strip().lower()
        if pkg:
            existing_reqs.add(pkg)

    # Common mapping from import name to package name
    import_to_package = {
        "jose": "python-jose",
        "jwt": "pyjwt",
        "dotenv": "python-dotenv",
        "bs4": "beautifulsoup4",
        "dateutil": "python-dateutil",
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "multipart": "python-multipart" # Common for fastapi upload
    }

    missing_packages = set()

    for filename, content in files.items():
        if filename.startswith("backend/") and filename.endswith(".py"):
            # Regex to find imports
            # import xyz
            # from xyz import abc
            # from xyz.def import abc
            
            imports = re.findall(r'^\s*import\s+([\w\.]+)', content, re.MULTILINE)
            from_imports = re.findall(r'^\s*from\s+([\w\.]+)\s+import', content, re.MULTILINE)
            
            all_imports = imports + from_imports
            for imp in all_imports:
                top_level = imp.split('.')[0]
                
                # Skip stdlib (heuristic: broad list or check against known stdlib? 
                # For now assume if it's common and not in requirements, it might be 3rd party.
                # But to avoid false positives on stdlib, we only check known missing ones or use a simple stdlib list)
                
                # Better approach: Just add specific known mappings if missing. 
                # or check if it's NOT in existing_reqs and MIGHT be a package.
                
                package_name = import_to_package.get(top_level, top_level.lower())
                
                # Filter out standard library (partial list)
                if top_level in [
                    "os", "sys", "json", "typing", "datetime", "time", "random", "math", "re", 
                    "collections", "itertools", "functools", "logging", "asyncio", "pathlib", 
                    "shutil", "subprocess", "uuid", "abc", "io", "copy", "hashlib", "base64",
                    "typing", "enum", "dataclasses", "contextlib"
                ]:
                    continue
                    
                # Also skip local modules (if a file exists matching the import)
                # check if backend/<top_level>.py exists
                if f"backend/{top_level}.py" in files:
                    continue
                    
                # Skip local modules (services, etc.) - not PyPI packages
                if package_name in ("services",):
                    continue
                if f"backend/{top_level}.py" in files:
                    continue
                if any(k.startswith(f"backend/{top_level}/") for k in files.keys()):
                    continue
                if package_name not in existing_reqs:
                    missing_packages.add(package_name)

    # Add missing packages (package names only, no version numbers)
    if missing_packages:
        new_reqs = [pkg for pkg in missing_packages if pkg not in existing_reqs]
        for pkg in new_reqs:
            existing_reqs.add(pkg)
        if new_reqs:
            files[req_path] = files[req_path].strip() + "\n" + "\n".join(new_reqs) + "\n"
    
    # Normalize: strip version numbers, remove erroneous packages (services=local module)
    REQ_BLOCKLIST = {"services", "motor"}
    lines = []
    seen = set()
    for raw in files[req_path].strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            lines.append(line)
            continue
        # Strip version: package, package==x, package>=x, package~=x, package[extra] -> package
        pkg = re.split(r'[=<>~!\[]', line)[0].strip().lower()
        if pkg and pkg not in seen and pkg not in REQ_BLOCKLIST:
            seen.add(pkg)
            lines.append(pkg)
    files[req_path] = "\n".join(lines) + "\n"


def _validate_frontend_dependencies(files: Dict[str, str]):
    """
    Scan frontend js/ts files for imports and verify package.json
    """
    pkg_path = "frontend/package.json"
    if pkg_path not in files:
        return

    try:
        pkg_json = json.loads(files[pkg_path])
        dependencies = pkg_json.get("dependencies", {})
        dev_dependencies = pkg_json.get("devDependencies", {})
        all_deps = set(dependencies.keys()) | set(dev_dependencies.keys())
    except:
        return

    # Common mappings
    import_to_pkg = {
        "react-router-dom": "react-router-dom",
        "framer-motion": "framer-motion",
        "lucide-react": "lucide-react",
        "axios": "axios",
        "clsx": "clsx",
        "tailwind-merge": "tailwind-merge",
        "date-fns": "date-fns",
        "@radix-ui/react-slot": "@radix-ui/react-slot"
        # Add more as needed
    }

    missing_deps = set()

    # 1. Check for missing dependencies based on imports
    for filename, content in files.items():
        if filename.startswith("frontend/src/") and filename.split('.')[-1] in ['js', 'jsx', 'ts', 'tsx']:
            matches = re.findall(r'from\s+[\'"]([@\w\-/]+)[\'"]', content)
            
            for module_name in matches:
                if module_name.startswith("."): # Relative import
                    continue
                
                pkg_check = module_name
                if module_name.startswith("@"):
                    parts = module_name.split("/")
                    if len(parts) >= 2:
                        pkg_check = f"{parts[0]}/{parts[1]}"
                else:
                    pkg_check = module_name.split("/")[0]

                if pkg_check == "react" or pkg_check == "react-dom":
                    continue 
                
                if pkg_check not in all_deps and pkg_check not in ["./", "../"] and pkg_check not in FRONTEND_BLOCKLIST:
                    missing_deps.add(pkg_check)

    # 2. Remove blocklisted packages (keep minimal - no Tailwind)
    for pkg in FRONTEND_BLOCKLIST:
        dependencies.pop(pkg, None)
        dev_dependencies.pop(pkg, None)

    # 3. Enforce Known Stable Versions (Fixes bad versions like ^0.1.0)
    for pkg, stable_ver in KNOWN_STABLE_VERSIONS.items():
        # If package is used (either already in deps or found missing)
        # We enforce the stable version.
        if pkg in all_deps or pkg in missing_deps:
            dependencies[pkg] = stable_ver
            # Remove from missing since we just handled it
            if pkg in missing_deps:
                missing_deps.remove(pkg)

    # 4. Add remaining missing deps as "latest" (excluding blocklist)
    if missing_deps:
        for dep in missing_deps:
            dependencies[dep] = "latest"
    
    pkg_json["dependencies"] = dependencies
    pkg_json["devDependencies"] = dev_dependencies
    files[pkg_path] = json.dumps(pkg_json, indent=2)

