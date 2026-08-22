"""
Verify generated frontend builds during codegen and auto-fix with LLM (Cursor-style loop).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, Optional, Tuple

from app_builder.schemas.files import GeneratedFiles
from app_builder.services.file_writer import file_writer
from app_builder.services.runtime_paths import resolve_project_root
from llm_helper import get_llm_for_user

from api.app_creator.cra_npm_patch import (
    ensure_cra_build_env_file,
    patch_package_json_in_files,
    prepare_frontend_dir_on_disk,
)
from api.app_creator.pipeline_helpers import npm_build_timeout, npm_install_timeout

logger = logging.getLogger("app_builder")


def _resolve_frontend_dir(project_root: str) -> Optional[str]:
    frontend = os.path.join(project_root, "frontend")
    if os.path.isfile(os.path.join(frontend, "package.json")):
        return frontend
    client = os.path.join(frontend, "client")
    if os.path.isfile(os.path.join(client, "package.json")):
        return client
    if os.path.isfile(os.path.join(project_root, "package.json")):
        return project_root
    return None


def _capture_output(result: subprocess.CompletedProcess, cwd: str) -> str:
    parts = []
    if result.stdout:
        parts.append(result.stdout.decode("utf-8", errors="replace"))
    if result.stderr:
        parts.append(result.stderr.decode("utf-8", errors="replace"))
    text = "\n".join(parts).strip()
    if len(text) > 12000:
        text = text[-12000:]
    return text or f"npm failed in {cwd} (exit {result.returncode})"


def run_frontend_build(project_name: str, install: bool = True) -> Tuple[Optional[str], Optional[str], str]:
    """
    npm install + npm run build locally.
    Returns (static_dir, error_message, log_output).
    """
    project_root = resolve_project_root(project_name)
    if not os.path.isdir(project_root):
        return None, "Project not found on disk", ""

    frontend_dir = _resolve_frontend_dir(project_root)
    if not frontend_dir:
        return None, "No frontend/package.json found", ""

    from api.app_creator.vite_build import is_vite_frontend, prepare_vite_frontend_dir, run_vite_build

    if is_vite_frontend(frontend_dir):
        prepare_vite_frontend_dir(frontend_dir)
        return run_vite_build(frontend_dir, project_name, install=install)

    patched = prepare_frontend_dir_on_disk(frontend_dir)
    if patched:
        node_modules = os.path.join(frontend_dir, "node_modules")
        if os.path.isdir(node_modules):
            shutil.rmtree(node_modules, ignore_errors=True)
    logs: list[str] = []

    try:
        if install:
            logs.append("[build] npm install --legacy-peer-deps")
            r = subprocess.run(
                ["npm", "install", "--legacy-peer-deps"],
                cwd=frontend_dir,
                capture_output=True,
                timeout=npm_install_timeout(),
            )
            out = _capture_output(r, frontend_dir)
            logs.append(out)
            if r.returncode != 0:
                return None, f"npm install failed (exit {r.returncode}): {out[-800:]}", "\n".join(logs)

        build_env = os.environ.copy()
        build_env["PUBLIC_URL"] = f"/app/{project_name}"
        from api.app_creator.cra_npm_patch import CRA_BUILD_ENV
        for key, val in CRA_BUILD_ENV.items():
            build_env.setdefault(key, val)

        logs.append("[build] npm run build")
        r = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            capture_output=True,
            timeout=npm_build_timeout(),
            env=build_env,
        )
        out = _capture_output(r, frontend_dir)
        logs.append(out)
        if r.returncode != 0:
            return None, f"npm run build failed (exit {r.returncode}): {out[-1200:]}", "\n".join(logs)

        for sub in ("dist", "build"):
            path = os.path.join(frontend_dir, sub)
            if os.path.isdir(path):
                logs.append(f"[build] OK → {sub}/")
                return path, None, "\n".join(logs)
        return None, "Build completed but dist/build not found", "\n".join(logs)
    except subprocess.TimeoutExpired:
        return None, "Build timed out", "\n".join(logs)
    except FileNotFoundError:
        return None, "npm not found — install Node.js on the server", "\n".join(logs)
    except Exception as exc:
        return None, str(exc), "\n".join(logs)


def _pick_files_for_fix(files: Dict[str, str], build_log: str) -> Dict[str, str]:
    """Select files most likely relevant to the build error."""
    priority = []
    if "package.json" in build_log or "ajv" in build_log.lower():
        priority.append("frontend/package.json")
    for path in sorted(files.keys()):
        if path in priority:
            continue
        if not path.startswith("frontend/"):
            continue
        if "node_modules" in path:
            continue
        if path.endswith((".js", ".jsx", ".ts", ".tsx", ".css", ".json", ".env")):
            priority.append(path)
    out: Dict[str, str] = {}
    for path in priority[:14]:
        if path in files:
            out[path] = files[path]
    if "frontend/package.json" not in out and "frontend/package.json" in files:
        out["frontend/package.json"] = files["frontend/package.json"]
    return out


async def _llm_fix_build(files: Dict[str, str], build_log: str, user_email: Optional[str]) -> Dict[str, str]:
    """Ask LLM to patch files to fix the build log."""
    subset = _pick_files_for_fix(files, build_log)
    snippets = []
    for path, content in subset.items():
        snippets.append(f"--- {path} ---\n{content[:4000]}")

    prompt = f"""You are fixing a Create React App (react-scripts 5) project that failed `npm run build`.

BUILD LOG (last lines):
```
{build_log[-6500:]}
```

CURRENT FILES:
{chr(10).join(snippets)}

Rules:
1. If the error mentions ajv, ajv-keywords, fork-ts-checker, or "reading 'date'":
   - Add devDependencies: @craco/craco 7.1.0, ajv 6.12.6, ajv-keywords 3.5.2
   - Set scripts.build to "NODE_OPTIONS=--openssl-legacy-provider craco build"
   - Set scripts.start to "NODE_OPTIONS=--openssl-legacy-provider craco start"
   - Add frontend/craco.config.js that filters out ForkTsCheckerWebpackPlugin from webpack plugins
   - Set overrides: {json.dumps({"ajv": "6.12.6", "ajv-keywords": "3.5.2"})}
2. Keep react-scripts 5; use NODE_OPTIONS=--openssl-legacy-provider in build script if missing.
3. Fix syntax/import errors in source files if shown in the log.
4. Return ONLY changed files as valid JSON: {{"files": {{"path/to/file": "full file content"}}}}
5. Do not truncate files — return complete file contents for each changed path."""

    llm = get_llm_for_user(user_email, temperature=0.15)
    resp = await llm.ainvoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(text, list):
        text = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in text
        )

    match = re.search(r"\{[\s\S]*\"files\"[\s\S]*\}", text)
    if not match:
        return files
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            data = json.loads(repair_json(match.group(0)))
        except Exception:
            return files

    updated = dict(files)
    for path, content in (data.get("files") or {}).items():
        if isinstance(content, str) and content.strip():
            updated[path] = content
    from app_builder.services.code_post_process import _is_vite_project
    if not _is_vite_project(updated):
        patch_package_json_in_files(updated)
        ensure_cra_build_env_file(updated)
    else:
        updated.pop("frontend/craco.config.js", None)
    return updated


async def verify_build_and_fix(
    project_name: str,
    files: Dict[str, str],
    user_email: Optional[str],
    on_event: Optional[Callable[[dict], Any]] = None,
    max_attempts: int = 3,
) -> Tuple[Dict[str, str], bool, str]:
    """
    Write files, run npm build, auto-fix with LLM on failure (up to max_attempts).
    Returns (updated_files, success, last_log).
    """
    async def emit(event: str, message: str, **extra):
        payload = {"event": event, "message": message, **extra}
        if on_event:
            await on_event(payload)

    from app_builder.services.code_post_process import _is_vite_project
    if not _is_vite_project(files):
        patch_package_json_in_files(files)
        ensure_cra_build_env_file(files)
    else:
        files.pop("frontend/craco.config.js", None)

    file_writer(project_name, GeneratedFiles(files=files))

    last_log = ""
    for attempt in range(1, max_attempts + 1):
        await emit(
            "build_start",
            f"Running npm install + build (attempt {attempt}/{max_attempts})...",
            attempt=attempt,
        )
        static_dir, err, log = run_frontend_build(project_name, install=True)
        last_log = log or err or ""
        if log:
            for line in log.splitlines()[-30:]:
                if line.strip():
                    await emit("build_log", line.strip())

        if static_dir and not err:
            await emit("build_success", "Frontend build verified successfully.")
            return files, True, last_log

        await emit(
            "build_failed",
            err or "Build failed",
            log=last_log[-2000:] if last_log else "",
            attempt=attempt,
        )

        if attempt >= max_attempts:
            break

        await emit(
            "build_fix_attempt",
            f"Build failed — AI is fixing errors (attempt {attempt}/{max_attempts})...",
            attempt=attempt,
        )
        try:
            files = await _llm_fix_build(files, last_log or (err or ""), user_email)
            file_writer(project_name, GeneratedFiles(files=files))
        except Exception as fix_err:
            logger.warning("[build_verify] LLM fix failed: %s", fix_err)
            await emit("build_fix_attempt", f"Auto-fix error: {fix_err}")

    return files, False, last_log
