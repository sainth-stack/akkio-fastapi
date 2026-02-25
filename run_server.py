#!/usr/bin/env python3
"""
Start the Akkio FastAPI server with reload. Use this instead of
`uvicorn final_akio_apis:app --reload` so that generated app-builder
code in app_builder/runtime/ does NOT trigger main server reloads.
"""
import os
import sys

# Ensure we're in the right directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)
sys.path.insert(0, _script_dir)

if __name__ == "__main__":
    import uvicorn

    from app_builder.services.runtime_paths import get_runtime_root

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # Exclude app_builder runtime dirs so generated app code never triggers main server reload
    _runtime_root = os.path.abspath(get_runtime_root())
    os.makedirs(_runtime_root, exist_ok=True)
    _candidates = [
        _runtime_root,
        os.path.join(_script_dir, "app_builder", ".runtime"),
        os.path.join(os.path.expanduser("~"), ".akkio", "app_builder", "runtime"),
        "app_builder/runtime",  # relative - watchfiles reports relative paths, exclude_dir must match path.parents
    ]
    _reload_excludes = list({d for d in _candidates if os.path.isdir(d)})

    uvicorn.run(
        "final_akio_apis:app",
        host=host,
        port=port,
        reload=True,
        reload_excludes=_reload_excludes,
    )
