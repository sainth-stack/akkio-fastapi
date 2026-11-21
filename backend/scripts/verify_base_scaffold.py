#!/usr/bin/env python3
"""Verify base-vite-fastapi scaffold loads and builds on Node 20+."""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_builder.services.scaffold_service import get_base_scaffold_files
from app_builder.services.code_post_process import post_process_generated_files


def main() -> int:
    files = post_process_generated_files(get_base_scaffold_files(), architecture={}, uiux="")
    print(f"✓ Loaded {len(files)} scaffold files")
    with tempfile.TemporaryDirectory() as tmp:
        for rel, content in files.items():
            full = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
        frontend = os.path.join(tmp, "frontend")
        print("→ npm install...")
        subprocess.run(["npm", "install", "--legacy-peer-deps"], cwd=frontend, check=True)
        env = os.environ.copy()
        env["VITE_BASE_PATH"] = "/app/verify/"
        print("→ npm run build...")
        subprocess.run(["npm", "run", "build"], cwd=frontend, check=True, env=env)
        dist = os.path.join(frontend, "dist")
        if not os.path.isdir(dist):
            print("✗ dist/ not found")
            return 1
        print(f"✓ Build OK → {dist}")
    print("\nBase Vite+FastAPI scaffold verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
