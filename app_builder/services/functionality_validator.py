"""
Functionality validator — ensures generated apps will work at runtime.

Checks and fixes:
- Wrong apiFetch(path, 'METHOD', body) signature (common LLM mistake)
- Frontend API paths aligned with project_config collection names
- project_config tables/fields populated from architecture
- SQLite CRUD smoke test (list / create / update / delete)
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("app_builder")

_FRONTEND_PATHS = (
    "frontend/src/App.jsx",
    "frontend/src/App.js",
    "frontend/App.js",
)


def fix_api_fetch_signature(content: str) -> Tuple[str, int]:
    """
    Fix apiFetch(path, 'METHOD') and apiFetch(path, 'METHOD', body)
    to use the options-object signature expected by api/client.js.
    """
    fixes = 0

    def _body_expr(raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("JSON.stringify("):
            return raw
        return f"JSON.stringify({raw})"

    def repl_with_body(m: re.Match) -> str:
        nonlocal fixes
        fixes += 1
        path, method, body = m.group(1), m.group(2).upper(), m.group(3).strip()
        return f"apiFetch({path}, {{ method: '{method}', body: {_body_expr(body)} }})"

    def repl_no_body(m: re.Match) -> str:
        nonlocal fixes
        fixes += 1
        path, method = m.group(1), m.group(2).upper()
        return f"apiFetch({path}, {{ method: '{method}' }})"

    content = re.sub(
        r"apiFetch\(([^,]+),\s*['\"](GET|POST|PUT|PATCH|DELETE)['\"]\s*,\s*([^)]+)\)",
        repl_with_body,
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(
        r"apiFetch\(([^,]+),\s*['\"](GET|POST|PUT|PATCH|DELETE)['\"]\s*\)",
        repl_no_body,
        content,
        flags=re.IGNORECASE,
    )
    return content, fixes


def _extract_api_paths(ui_src: str) -> List[str]:
    """Collect first path segment from apiFetch('/collection/...') calls."""
    paths: List[str] = []
    for m in re.finditer(r"apiFetch\(\s*(['\"`])(/[^'\"`]+)\1", ui_src):
        seg = m.group(2).strip("/").split("/")[0]
        if seg and seg not in paths:
            paths.append(seg)
    for m in re.finditer(r"apiFetch\(\s*`(/[^`]+)`", ui_src):
        seg = m.group(1).strip("/").split("/")[0]
        if seg and not seg.startswith("${") and seg not in paths:
            paths.append(seg)
    const_m = re.search(r"const\s+API\s*=\s*['\"`](/[^'\"`]+)['\"`]", ui_src)
    if const_m:
        seg = const_m.group(1).strip("/").split("/")[0]
        if seg and seg not in paths:
            paths.append(seg)
    return paths


def fields_from_architecture_tables(tables: List[Any]) -> Dict[str, List[str]]:
    fields: Dict[str, List[str]] = {}
    for t in tables or []:
        if not isinstance(t, dict):
            continue
        name = (t.get("name") or t.get("table_name") or "").strip()
        if not name:
            continue
        cols = t.get("columns") or t.get("fields") or []
        names: List[str] = []
        for c in cols:
            if isinstance(c, dict):
                n = (c.get("name") or "").strip()
            else:
                n = str(c).strip()
            if n and n.lower() != "id":
                names.append(n)
        if names:
            fields[name] = names
    return fields


def tables_from_architecture(architecture: Dict[str, Any]) -> List[str]:
    tables_raw = (architecture or {}).get("database_schema", {}).get("tables") or []
    names: List[str] = []
    for t in tables_raw:
        if isinstance(t, dict):
            n = (t.get("name") or t.get("table_name") or "").strip()
            if n:
                names.append(n)
        elif isinstance(t, str) and t.strip():
            names.append(t.strip())
    return names


def enrich_project_config(
    config: Dict[str, Any],
    architecture: Dict[str, Any],
    app_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ensure project_config has tables and fields for dynamic CRUD router."""
    out = dict(config)
    arch_tables = tables_from_architecture(architecture)
    spec_table = (app_spec or {}).get("primary_table")

    if not out.get("tables"):
        if arch_tables:
            out["tables"] = arch_tables
        elif spec_table:
            out["tables"] = [spec_table]

    fields = dict(out.get("fields") or {})
    if not fields:
        fields.update(fields_from_architecture_tables(
            (architecture or {}).get("database_schema", {}).get("tables") or []
        ))
    if not fields and spec_table:
        spec_fields = (app_spec or {}).get("fields") or []
        if spec_fields:
            fields[spec_table] = [f for f in spec_fields if f != "id"]
        elif spec_table in ("tasks", "todos", "items"):
            fields[spec_table] = ["title", "completed"]
        else:
            fields[spec_table] = ["title"]
    out["fields"] = fields
    return out


def align_frontend_collection(
    files: Dict[str, str],
    collection: str,
) -> Tuple[Dict[str, str], int]:
    """Rewrite frontend API paths to use the runtime collection name (e.g. tasks)."""
    if not collection:
        return files, 0
    out = dict(files)
    fixes = 0
    target = f"/{collection.lstrip('/')}"

    for path in _FRONTEND_PATHS:
        if path not in out:
            continue
        src = out[path]
        original = src

        src = re.sub(
            r"const\s+API\s*=\s*['\"`]/[^'\"`]+['\"`]",
            f"const API = '{target}'",
            src,
            count=1,
        )

        wrong_collections = ("items", "todos", "tasks", "entries", "lists", "trips")
        for wrong in wrong_collections:
            if wrong == collection.lstrip("/"):
                continue
            src = re.sub(rf"apiFetch\(\s*['\"]/{wrong}(['\"/])", rf"apiFetch('{target}\1", src)
            src = re.sub(rf"apiFetch\(\s*`/{wrong}([/`])", rf"apiFetch(`{target}\1", src)

        if src != original:
            fixes += 1
            out[path] = src

    for path in list(out.keys()):
        if not path.startswith("frontend/src/components/"):
            continue
        src = out[path]
        new_src, n = fix_api_fetch_signature(src)
        if n:
            fixes += n
            out[path] = new_src

    return out, fixes


def detect_api_fetch_issues(ui_src: str) -> List[str]:
    issues: List[str] = []
    if re.search(
        r"apiFetch\([^,]+,\s*['\"](GET|POST|PUT|PATCH|DELETE)['\"]",
        ui_src,
        re.I,
    ):
        issues.append("apiFetch uses wrong signature (method string instead of options object)")
    paths = _extract_api_paths(ui_src)
    if not paths and "apiFetch" in ui_src:
        issues.append("apiFetch used but no collection path detected")
    return issues


def ensure_crud_create_input(ui_src: str, app_spec: Optional[Dict[str, Any]] = None) -> Tuple[str, bool]:
    """
    Ensure CRUD todo-style apps have an input bound to the create-task state.
    LLM often generates a create button without a title field.
    """
    kind = (app_spec or {}).get("app_kind", "")
    if kind not in ("crud", "custom", "dashboard"):
        return ui_src, False

    has_create = any(
        k in ui_src
        for k in ("handleCreateTask", "addItem", "handleCreate", "handleAdd", "createTask")
    )
    if not has_create:
        return ui_src, False

    if re.search(r'onChange=\{[^}]*(setNewTask|setTitle)[^}]*title', ui_src):
        return ui_src, False
    if re.search(r'value=\{(newTask\.title|title)\}', ui_src) and "placeholder" in ui_src.lower():
        return ui_src, False

    field = "newTask.title" if "setNewTask" in ui_src else "title"
    setter = "setNewTask({ ...newTask, title: e.target.value })" if "setNewTask" in ui_src else "setTitle(e.target.value)"
    input_block = f'''
        <input
          type="text"
          placeholder="Add a new task..."
          value={{{field}}}
          onChange={{(e) => {setter}}}
          aria-label="New task title"
        />'''

    if "<button onClick={handleCreateTask}" in ui_src:
        ui_src = ui_src.replace(
            "<button onClick={handleCreateTask}",
            input_block + "\n        <button onClick={handleCreateTask}",
            1,
        )
        return ui_src, True
    if "<form" not in ui_src.lower() and "addItem" in ui_src:
        return ui_src, False
    return ui_src, False


def validate_and_fix_functionality(
    files: Dict[str, str],
    architecture: Dict[str, Any],
    app_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Fix common runtime-breaking patterns in generated frontend code.
    Returns (files, remaining_errors, fix_messages).
    """
    out = dict(files)
    fix_messages: List[str] = []
    errors: List[str] = []

    collection = (
        (app_spec or {}).get("primary_table")
        or (tables_from_architecture(architecture)[0] if tables_from_architecture(architecture) else "items")
    )

    for path in _FRONTEND_PATHS:
        if path not in out:
            continue
        fixed, n = fix_api_fetch_signature(out[path])
        if n:
            fix_messages.append(f"Fixed {n} apiFetch call(s) in {path}")
            out[path] = fixed
        errors.extend(detect_api_fetch_issues(out[path]))

        patched, added_input = ensure_crud_create_input(out[path], app_spec)
        if added_input:
            fix_messages.append(f"Added missing task title input in {path}")
            out[path] = patched

    out, align_n = align_frontend_collection(out, collection)
    if align_n:
        fix_messages.append(f"Aligned API paths to collection '/{collection}' ({align_n} file(s))")

    errors = list(dict.fromkeys(errors))
    return out, errors, fix_messages


def _ensure_table(conn: sqlite3.Connection, table: str, fields: List[str]) -> None:
    col_defs = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
    for col in fields:
        col_defs.append(f'"{col}" TEXT')
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(col_defs)})')
    conn.commit()
    existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()}
    for col in fields:
        if col not in existing:
            try:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" TEXT')
                conn.commit()
            except sqlite3.OperationalError:
                pass


def verify_runtime_crud(
    project_id: str,
    config: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Smoke-test SQLite CRUD for the first configured collection.
    Does not require HTTP/auth — validates the data layer works.
    """
    from app_builder.services.runtime_paths import resolve_project_root

    tables = config.get("tables") or []
    if not tables:
        return False, "project_config has no tables"

    collection = tables[0]
    fields = config.get("fields") or {}
    col_fields = fields.get(collection) or ["title"]
    payload = {col_fields[0]: "__smoke_test__"}
    if "completed" in col_fields:
        payload["completed"] = False

    project_root = resolve_project_root(project_id)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "app.db")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        _ensure_table(conn, collection, col_fields)
        try:
            col_names = [k for k in payload.keys() if k != "id"]
            vals = [str(payload[k]) for k in col_names]
            placeholders = ",".join("?" * len(col_names))
            col_str = ",".join(f'"{c}"' for c in col_names)
            conn.execute(
                f'INSERT INTO "{collection}" ({col_str}) VALUES ({placeholders})',
                vals,
            )
            conn.commit()
            rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            cur = conn.execute(f'SELECT * FROM "{collection}" WHERE id = ?', (rid,))
            row = cur.fetchone()
            if not row:
                return False, f"CREATE failed: row {rid} not found"

            if "completed" in col_fields:
                conn.execute(
                    f'UPDATE "{collection}" SET "completed" = ? WHERE id = ?',
                    ("1", rid),
                )
                conn.commit()

            conn.execute(f'DELETE FROM "{collection}" WHERE id = ?', (rid,))
            conn.commit()
            return True, f"CRUD smoke test passed for '{collection}'"
        finally:
            conn.close()
    except Exception as e:
        logger.warning("[functionality] CRUD smoke test failed: %s", e)
        return False, str(e)


def run_functionality_pipeline(
    files: Dict[str, str],
    project_name: str,
    architecture: Dict[str, Any],
    app_spec: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], bool, List[str], str]:
    """
    Full functionality validation: fix code, enrich config, smoke-test CRUD.
    Returns (files, ok, messages, error).
    """
    messages: List[str] = []
    files, issues, fixes = validate_and_fix_functionality(files, architecture, app_spec)
    messages.extend(fixes)

    if issues:
        return files, False, messages, "; ".join(issues)

    if config:
        config = enrich_project_config(config, architecture, app_spec)
        from app_builder.services.project_config import persist_project_config
        from app_builder.services.db_init import ensure_project_db_initialized

        persist_project_config(project_name, config)
        ensure_project_db_initialized(project_name, config)
        ok, msg = verify_runtime_crud(project_name, config)
        messages.append(msg)
        if not ok:
            return files, False, messages, msg

    return files, True, messages, ""
