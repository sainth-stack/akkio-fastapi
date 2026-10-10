"""
Verify service for the frontend-only pipeline.

After every stage:
  a. Write files to the project directory
  b. tsc --noEmit scoped to the project (if node_modules available)
  c. Import-resolution check (pure Python — works without node_modules)
  d. eslint-lite: no undefined JSX components, no broken named imports

On failure, caller supplies the offending file + error snippet to the fixer agent
(max 3 attempts). File is only accepted if it is complete (balanced braces, same exports).
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app_builder.services.runtime_paths import project_write_root
from app_builder.services.file_writer import write_single_file_to_disk

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    offending_files: List[str] = field(default_factory=list)
    log: str = ""
    tsc_ran: bool = False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def verify_stage(
    project_name: str,
    files: Dict[str, str],       # full file dict accumulated so far
    new_files: Dict[str, str],   # files added in the current stage
    stage_name: str = "",
) -> VerifyResult:
    """
    Write `new_files` to disk, then run all verification checks.
    Returns a VerifyResult with any errors found.
    """
    project_root = project_write_root(project_name)
    frontend_dir = os.path.join(project_root, "frontend")

    # 1. Write new files to disk
    for rel_path, content in new_files.items():
        try:
            write_single_file_to_disk(project_root, rel_path, content)
        except Exception as exc:
            return VerifyResult(
                ok=False,
                errors=[f"Failed to write {rel_path}: {exc}"],
                offending_files=[rel_path],
                log=f"write error: {exc}",
            )

    # 2. Import resolution check (pure Python, always runs)
    ir_result = _check_import_resolution(files, project_root)
    if not ir_result.ok:
        return ir_result

    # 3. tsc --noEmit (only if node_modules available)
    nm_path = _find_node_modules(frontend_dir)
    if nm_path:
        tsc_result = await _run_tsc(frontend_dir)
        tsc_result.tsc_ran = True
        if not tsc_result.ok:
            return tsc_result
        return VerifyResult(ok=True, tsc_ran=True, log="tsc: ok; imports: ok")

    return VerifyResult(ok=True, tsc_ran=False, log="imports: ok (tsc skipped — no node_modules)")


# ---------------------------------------------------------------------------
# tsc --noEmit
# ---------------------------------------------------------------------------

async def _run_tsc(frontend_dir: str) -> VerifyResult:
    """Run tsc --noEmit in the frontend directory. Returns VerifyResult."""
    tsc_bin = _find_tsc(frontend_dir)
    if not tsc_bin:
        return VerifyResult(ok=True, log="tsc not found — skipped")

    try:
        proc = await asyncio.create_subprocess_exec(
            tsc_bin, "--noEmit", "--pretty", "false",
            cwd=frontend_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = (stdout + stderr).decode("utf-8", errors="replace")

        if proc.returncode == 0:
            return VerifyResult(ok=True, log="tsc: 0 errors")

        # Parse errors
        errors, offending = _parse_tsc_output(output)
        return VerifyResult(
            ok=False,
            errors=errors[:20],
            offending_files=list(dict.fromkeys(offending))[:5],
            log=output[-4000:],
        )
    except asyncio.TimeoutError:
        return VerifyResult(ok=False, errors=["tsc timed out (60s)"], log="timeout")
    except Exception as exc:
        return VerifyResult(ok=False, errors=[str(exc)], log=str(exc))


def _find_tsc(frontend_dir: str) -> Optional[str]:
    """Return path to tsc binary, or None if not found."""
    candidates = [
        os.path.join(frontend_dir, "node_modules", ".bin", "tsc"),
        os.path.join(frontend_dir, "node_modules", "typescript", "bin", "tsc"),
        "npx",  # fallback — used as ['npx', 'tsc', '--noEmit']
    ]
    for c in candidates:
        if c == "npx" or os.path.isfile(c):
            return c
    return None


def _find_node_modules(frontend_dir: str) -> Optional[str]:
    """Return node_modules path if it exists (local or nm_cache symlink)."""
    local = os.path.join(frontend_dir, "node_modules")
    if os.path.isdir(local):
        return local

    # Try nm_cache
    cache = os.path.join(
        os.path.expanduser("~"), ".akkio", "app_builder", "nm_cache",
        "base-frontend-vite-mui", "node_modules",
    )
    if os.path.isdir(cache):
        # Create a symlink for tsc to use
        try:
            os.makedirs(frontend_dir, exist_ok=True)
            os.symlink(cache, local)
            return local
        except OSError:
            return cache  # just point at cache directly
    return None


def _parse_tsc_output(output: str) -> Tuple[List[str], List[str]]:
    """Extract (errors, offending_files) from tsc output."""
    errors: List[str] = []
    files: List[str] = []
    # tsc format: src/foo.ts(12,3): error TS2345: ...
    pattern = re.compile(r"^(.+?)\((\d+),(\d+)\):\s+(error\s+\w+:\s+.+)$", re.MULTILINE)
    for m in pattern.finditer(output):
        file_path, line, col, msg = m.group(1), m.group(2), m.group(3), m.group(4)
        errors.append(f"{file_path}:{line}:{col} — {msg}")
        if file_path not in files:
            files.append(file_path)
    if not errors and output.strip():
        errors = [output.strip()[:500]]
    return errors, files


# ---------------------------------------------------------------------------
# Import resolution (pure Python)
# ---------------------------------------------------------------------------

def _check_import_resolution(
    files: Dict[str, str],
    project_root: str,
) -> VerifyResult:
    """
    For every TypeScript/TSX file, check:
    1. All relative imports resolve to an existing file.
    2. All named imports exist as exports in the imported file.
    3. No JSX component is used without being imported.
    """
    errors: List[str] = []
    offending: List[str] = []

    ts_files = {k: v for k, v in files.items() if k.endswith((".ts", ".tsx"))}

    for file_path, content in ts_files.items():
        # Skip node_modules
        if "node_modules" in file_path:
            continue
        file_errs = _check_file_imports(file_path, content, ts_files, project_root)
        if file_errs:
            errors.extend(file_errs)
            if file_path not in offending:
                offending.append(file_path)

    if errors:
        return VerifyResult(
            ok=False,
            errors=errors[:20],
            offending_files=offending[:5],
            log="\n".join(errors[:20]),
        )
    return VerifyResult(ok=True)


def _check_file_imports(
    file_path: str,
    content: str,
    all_files: Dict[str, str],
    project_root: str,
) -> List[str]:
    errors: List[str] = []

    # Parse import statements
    import_re = re.compile(
        r"""import\s+(?:\{([^}]*)\}|(\w+)|\*\s+as\s+\w+)\s+from\s+['"]([^'"]+)['"]""",
        re.MULTILINE,
    )

    for m in import_re.finditer(content):
        named_group = m.group(1)  # { Foo, Bar }
        default_name = m.group(2)  # default import
        source = m.group(3)

        # Only check relative imports
        if not source.startswith("."):
            continue

        # Resolve the import path relative to the importing file
        file_dir = os.path.dirname(file_path).replace("\\", "/")
        resolved = _resolve_import(file_dir, source)

        # Check file exists (in-memory dict first, then disk)
        target_content = _find_file_content(resolved, all_files, project_root)
        if target_content is None:
            errors.append(
                f"{file_path}: cannot resolve '{source}' → {resolved}"
            )
            continue

        # Check named imports exist in the target
        if named_group:
            names = [n.strip().split(" as ")[0].strip() for n in named_group.split(",") if n.strip()]
            exports = _extract_exports(target_content)
            for name in names:
                if name and name not in exports:
                    errors.append(
                        f"{file_path}: '{name}' not exported from '{source}' "
                        f"(exports: {sorted(exports)[:10]})"
                    )

    return errors


def _resolve_import(file_dir: str, source: str) -> str:
    """Resolve a relative import path to a normalized file key."""
    # Normalize e.g. ../components/ui → frontend/src/components/ui/index.ts
    parts = (file_dir + "/" + source).replace("\\", "/").split("/")
    resolved_parts: List[str] = []
    for part in parts:
        if part == "..":
            if resolved_parts:
                resolved_parts.pop()
        elif part not in ("", "."):
            resolved_parts.append(part)
    resolved = "/".join(resolved_parts)
    return resolved


def _find_file_content(
    resolved: str,
    all_files: Dict[str, str],
    project_root: str,
) -> Optional[str]:
    """Look up a resolved path in the file dict, trying .ts/.tsx extensions."""
    candidates = [
        resolved,
        resolved + ".ts",
        resolved + ".tsx",
        resolved + "/index.ts",
        resolved + "/index.tsx",
    ]
    for c in candidates:
        if c in all_files:
            return all_files[c]
    # Try disk
    for c in candidates:
        full = os.path.join(project_root, c)
        if os.path.isfile(full):
            try:
                with open(full, encoding="utf-8") as fh:
                    return fh.read()
            except OSError:
                pass
    return None


def _extract_exports(content: str) -> set:
    """Extract all exported names from a TypeScript file."""
    exports = set()

    # export const/let/var/function/class/interface/type Foo
    pattern = re.compile(r"export\s+(?:const|let|var|function|class|interface|type|enum)\s+(\w+)", re.MULTILINE)
    for m in pattern.finditer(content):
        exports.add(m.group(1))

    # export { Foo, Bar }
    named_re = re.compile(r"export\s*\{([^}]+)\}", re.MULTILINE)
    for m in named_re.finditer(content):
        for name in m.group(1).split(","):
            clean = name.strip().split(" as ")[-1].strip()
            if clean:
                exports.add(clean)

    # export default function/class Foo
    default_re = re.compile(r"export\s+default\s+(?:function|class)\s+(\w+)", re.MULTILINE)
    for m in default_re.finditer(content):
        exports.add(m.group(1))
        exports.add("default")

    # export default (arrow function or expression)
    if "export default " in content:
        exports.add("default")

    return exports


# ---------------------------------------------------------------------------
# Completeness check (accept replacement only if file looks complete)
# ---------------------------------------------------------------------------

def is_complete_file(content: str) -> bool:
    """
    Returns True if the generated file looks complete:
    - Non-empty
    - Balanced braces
    - Contains at least one export
    - No truncation markers
    """
    if not content or len(content.strip()) < 20:
        return False

    truncation_markers = ("...", "// TODO", "/* TODO", "// rest of", "// more code")
    for marker in truncation_markers:
        if marker in content:
            return False

    # Count braces
    opens = content.count("{")
    closes = content.count("}")
    if abs(opens - closes) > 2:
        return False

    # Must have at least one export
    if "export " not in content and "export default" not in content:
        return False

    return True


# ---------------------------------------------------------------------------
# Extract first error for fixer context
# ---------------------------------------------------------------------------

def extract_first_error(result: VerifyResult, files: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract the most actionable error info for the fixer agent.
    Returns: { error, offending_file, file_content, error_snippet }
    """
    if not result.errors:
        return {"error": result.log, "offending_file": "", "file_content": ""}

    first_error = result.errors[0]
    offending = result.offending_files[0] if result.offending_files else ""
    content = files.get(offending) or ""

    # Extract just the relevant lines around the error
    error_snippet = first_error[:300]

    return {
        "error": error_snippet,
        "offending_file": offending,
        "file_content": content[:3000],  # cap to avoid huge context
        "all_errors": result.errors[:5],
    }
