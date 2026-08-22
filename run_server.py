#!/usr/bin/env python3
"""
Start the Akkio FastAPI server with reload. Use this instead of
`uvicorn main:app --reload` so that generated app-builder
code in app_builder/runtime/ does NOT trigger main server reloads.
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
    # uvicorn only accepts relative glob patterns (absolute paths crash on Python 3.9).
    _reload_excludes = [
        "app_builder/runtime",
        "app_builder/runtime/*",
        "app_builder/.runtime",
        "app_builder/.runtime/*",
    ]

    print(
        "\n*** Akkio dev server (reload excludes generated app runtime) ***\n"
        f"    runtime: {_runtime_root}\n"
        f"    reload excludes: {_reload_excludes}\n"
        "    Tip: use this script instead of `uvicorn main:app --reload`.\n",
        file=sys.stderr,
    )

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        reload_excludes=_reload_excludes,
    )
