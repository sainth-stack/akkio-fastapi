import re
import json
from typing import Dict, Any, List

from app_builder.agents.dynamic_code_generator import (
    _fix_frontend_map_safety,
    _fix_frontend_backend_url_undefined,
    _fix_backend_routes_import,
)


def validate_and_fix_code(files: Dict[str, str], architecture: Dict[str, Any]) -> Dict[str, str]:
    """
    Validates and attempts to fix the generated code.
    1. Checks imports in backend/*.py and ensures they are in backend/requirements.txt
    2. Checks imports in frontend/src/** and ensures they are in frontend/package.json
    3. Verifies existence of critical files.
    4. Fixes frontend .map() safety (todos.map -> (todos||[]).map) to prevent "map is not a function"
    5. Fixes backend routes import (from routes import todo_router) to prevent ImportError
    """
    
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
    
    return files

# Minimal frontend - no Tailwind, no heavy deps. Only add if explicitly needed.
KNOWN_STABLE_VERSIONS = {
    "react-router-dom": "^6.16.0",
    "axios": "^1.5.0",
}

# Packages to never add - use styles.css instead, keep minimal
FRONTEND_BLOCKLIST = {"tailwindcss", "postcss", "autoprefixer", "framer-motion", "lucide-react", "clsx", "tailwind-merge"}

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
    REQ_BLOCKLIST = {"services"}
    lines = []
    seen = set()
    for raw in files[req_path].strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            lines.append(line)
            continue
        pkg = line.split("==")[0].split(">=")[0].split("[")[0].strip().lower()
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

