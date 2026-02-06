import re
import json
from typing import Dict, Any, List

def validate_and_fix_code(files: Dict[str, str], architecture: Dict[str, Any]) -> Dict[str, str]:
    """
    Validates and attempts to fix the generated code.
    1. Checks imports in backend/*.py and ensures they are in backend/requirements.txt
    2. Checks imports in frontend/src/** and ensures they are in frontend/package.json
    3. Verifies existence of critical files.
    """
    
    # 1. Backend Validation
    _validate_backend_dependencies(files)
    
    # 2. Frontend Validation
    _validate_frontend_dependencies(files)
    
    return files

# Hardcoded stable versions to prevent "notarget" errors from hallucinations (e.g. lucide-react@0.1.0)
KNOWN_STABLE_VERSIONS = {
    "lucide-react": "^0.263.1",
    "framer-motion": "^10.16.4",
    "clsx": "^2.0.0",
    "tailwind-merge": "^1.14.0",
    "react-router-dom": "^6.16.0",
    "date-fns": "^2.30.0",
    "axios": "^1.5.0",
    "@radix-ui/react-slot": "^1.0.2",
    "class-variance-authority": "^0.7.0"
}

def _validate_backend_dependencies(files: Dict[str, str]):
    """
    Scan backend python files for imports and ensure they are in requirements.txt
    """
    req_path = "backend/requirements.txt"
    if req_path not in files:
        # If requirements doesn't exist, create it (shouldn't happen with correct gen, but safety net)
        files[req_path] = "fastapi\nuvicorn\nsqlalchemy\npsycopg2-binary\npydantic\n"
    
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
                    
                if package_name not in existing_reqs:
                    # Heuristic: verify if it looks like a package
                    missing_packages.add(package_name)

    # Add missing packages
    if missing_packages:
        new_reqs = []
        for pkg in missing_packages:
            # Simple addition, no version pinning (could improve later)
            new_reqs.append(f"{pkg}")
            existing_reqs.add(pkg)
            
        if new_reqs:
            files[req_path] = files[req_path].strip() + "\n" + "\n".join(new_reqs) + "\n"


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
                
                if pkg_check not in all_deps and pkg_check not in ["./", "../"]:
                    missing_deps.add(pkg_check)

    # 2. Enforce Known Stable Versions (Fixes bad versions like ^0.1.0)
    for pkg, stable_ver in KNOWN_STABLE_VERSIONS.items():
        # If package is used (either already in deps or found missing)
        # We enforce the stable version.
        if pkg in all_deps or pkg in missing_deps:
            dependencies[pkg] = stable_ver
            # Remove from missing since we just handled it
            if pkg in missing_deps:
                missing_deps.remove(pkg)

    # 3. Add remaining missing deps as "latest"
    if missing_deps:
        for dep in missing_deps:
            dependencies[dep] = "latest"
    
    # Always save if we touched dependencies (which we likely did with Enforce Stable)
    # But only if changes actually happened. To be safe, just write it.
    pkg_json["dependencies"] = dependencies
    files[pkg_path] = json.dumps(pkg_json, indent=2)

