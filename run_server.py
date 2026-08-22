#!/usr/bin/env python3
"""
Start the Akkio FastAPI server with reload. Use this instead of
`uvicorn main:app --reload` so that generated app-builder
code in app_builder/runtime/ does NOT trigger main server reloads.

Important: uvicorn's reload_excludes use relative Path checks that fail against
absolute watched paths, so we whitelist reload_dirs instead of excluding runtime.
"""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)
sys.path.insert(0, _script_dir)

if __name__ == "__main__":
    import uvicorn

    from app_builder.services.runtime_paths import get_runtime_root

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    _runtime_root = os.path.abspath(get_runtime_root())
    os.makedirs(_runtime_root, exist_ok=True)

    # Only watch source packages — never app_builder/runtime (generated apps / npm).
    _reload_dirs = [
        os.path.join(_script_dir, "api"),
        os.path.join(_script_dir, "db"),
        os.path.join(_script_dir, "app_builder", "agents"),
        os.path.join(_script_dir, "app_builder", "services"),
        os.path.join(_script_dir, "app_builder", "graph"),
        os.path.join(_script_dir, "app_builder", "schemas"),
        os.path.join(_script_dir, "app_builder", "templates"),
    ]
    _reload_dirs = [d for d in _reload_dirs if os.path.isdir(d)]

    print(
        "\n*** Akkio dev server ***\n"
        f"    runtime: {_runtime_root}\n"
        f"    reload_dirs: {[os.path.relpath(d, _script_dir) for d in _reload_dirs]}\n"
        "    (app_builder/runtime is NOT watched — codegen/npm won't kill WebSockets)\n"
        "    Tip: use this script instead of `uvicorn main:app --reload`.\n",
        file=sys.stderr,
    )

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        reload_dirs=_reload_dirs,
        reload_includes=["*.py"],
    )
