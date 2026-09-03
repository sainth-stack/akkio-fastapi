"""
Dynamic CRUD and LLM proxy for generated apps.
Single backend serves all projects via config-driven routes.
"""
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.auth.dependencies import CurrentUser
from api.auth.request_auth import resolve_user
from api.app_creator.project_access import assert_project_access

from app_builder.services.project_config import get_project_config
from app_builder.services.runtime_paths import get_projects_dir, resolve_project_root

logger = logging.getLogger("app_builder")

router = APIRouter(prefix="/api/apps", tags=["Dynamic Apps"])


def _get_project_db_path(project_id: str) -> str:
    """Path to per-project SQLite DB."""
    project_root = resolve_project_root(project_id)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "app.db")


def _extract_fields_from_contract(api_contract: Dict[str, Any]) -> Dict[str, List[str]]:
    """Derive field names per collection from api_contract (used when config.fields is empty)."""
    fields: Dict[str, List[str]] = {}
    endpoints = api_contract.get("endpoints") or {}
    for path_spec, spec in endpoints.items():
        parts = path_spec.split()
        if len(parts) < 2:
            continue
        path = parts[1].strip("/")
        segment = path.split("/")[0]
        if not segment or segment.isdigit() or segment == "api":
            continue
        response = spec.get("response") or {}
        if isinstance(response, dict):
            field_names = [k for k in response.keys() if k != "id"]
        elif isinstance(response, list) and response and isinstance(response[0], dict):
            field_names = [k for k in response[0].keys() if k != "id"]
        else:
            continue
        if field_names and (segment not in fields or len(field_names) > len(fields.get(segment, []))):
            fields[segment] = field_names
    return fields


def _build_schema_from_config(config: Dict[str, Any]) -> str:
    """Generate CREATE TABLE SQL from config tables and fields. Works for any project (todos, users, orders, etc.)."""
    tables = config.get("tables") or []
    fields = config.get("fields") or {}
    if not fields and config.get("api_contract"):
        fields = _extract_fields_from_contract(config["api_contract"])
    statements = []
    for table in tables:
        cols = fields.get(table, [])
        col_defs = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
        for col in cols:
            col_defs.append(f'"{col}" TEXT')
        statements.append(f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(col_defs)})')
    return "; ".join(statements)


def _ensure_schema(conn: sqlite3.Connection, db_schema: str, config: Dict[str, Any]) -> None:
    """Create tables from db_schema if valid SQL, else from config tables/fields."""
    schema_sql = ""
    if db_schema and "CREATE TABLE" in db_schema.upper():
        schema_sql = db_schema
    if not schema_sql:
        schema_sql = _build_schema_from_config(config)
    if not schema_sql.strip():
        return
    statements = [
        s.strip() for s in schema_sql.split(";") if s.strip() and not s.strip().startswith("--")
    ]
    for stmt in statements:
        if not stmt.upper().startswith("CREATE"):
            continue
        try:
            conn.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError as e:
            if "already exists" not in str(e).lower():
                logger.warning("[dynamic_app] schema stmt failed: %s", e)


def _get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    """Return existing column names for a table. Empty if table does not exist."""
    try:
        cursor = conn.execute(f'PRAGMA table_info("{table}")')
        return [row[1] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []


def _get_expected_columns(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Get expected columns per table from config (fields + api_contract fallback)."""
    tables = config.get("tables") or []
    fields = config.get("fields") or {}
    if not fields and config.get("api_contract"):
        fields = _extract_fields_from_contract(config["api_contract"])
    return {t: fields.get(t, []) for t in tables}


def _ensure_columns(conn: sqlite3.Connection, table: str, expected_cols: List[str]) -> None:
    """Add any missing columns to existing table. Handles schema evolution (e.g. new 'priority' field)."""
    existing = _get_table_columns(conn, table)
    for col in expected_cols:
        if col not in existing:
            try:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" TEXT')
                conn.commit()
                logger.info("[dynamic_app] added column %s to %s", col, table)
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    logger.warning("[dynamic_app] add column failed: %s", e)


def _migrate_schema(conn: sqlite3.Connection, config: Dict[str, Any], extra_cols: Optional[Dict[str, List[str]]] = None) -> None:
    """Ensure tables exist and have all expected columns. Adds missing columns from config or request body."""
    expected = _get_expected_columns(config)
    if extra_cols:
        for table, cols in extra_cols.items():
            expected.setdefault(table, []).extend(c for c in cols if c not in expected.get(table, []))
    for table, cols in expected.items():
        _ensure_columns(conn, table, cols)


def _get_connection(project_id: str, config: Dict[str, Any], extra_cols: Optional[Dict[str, List[str]]] = None) -> sqlite3.Connection:
    """Get SQLite connection for project, ensuring schema exists and is migrated (add missing columns)."""
    db_path = _get_project_db_path(project_id)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    schema = config.get("db_schema") or ""
    _ensure_schema(conn, schema, config)
    _migrate_schema(conn, config, extra_cols)
    return conn


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row) if row else {}


# --- Custom endpoint handlers (dynamic for all app types) ---

GEN_PROMPTS = {
    "ideas": "You are a creative ideas generator. Given a topic, produce exactly {count} concise, actionable ideas. Return each idea on a new line, numbered 1. 2. 3. etc. No extra text.",
    "linkedin_post": "You are a LinkedIn content expert. Given a topic, write a professional, engaging LinkedIn post. Use a strong hook, short paragraphs (2-3 lines), end with a question or CTA. Include 3-5 relevant hashtags at the end.",
    "travel": "You are an expert travel planner. Given a destination, duration, or travel interest, produce a detailed travel plan with exactly {count} sections: overview, day-by-day itinerary ideas, must-see places, food recommendations, and practical tips. Use clear numbered sections.",
    "summarize": "You are an expert summarizer. Summarize the following text clearly and concisely. Preserve all key points, names, and conclusions. Use short paragraphs or bullet points. Return only the summary — no preamble or meta commentary.",
    "summarization": "You are an expert summarizer. Summarize the following text clearly and concisely. Preserve all key points. Return only the summary.",
    "translate": "You are a professional translator. Translate the given text accurately. Return only the translation, no explanations.",
    "general": "You are a helpful AI assistant. Given a topic or prompt, produce a thorough, useful response. For list-style requests, return exactly {count} items, each on a new line, numbered.",
}


def _infer_handler_from_path(path: str) -> str:
    """Infer handler type from endpoint path. Used when building config from api_contract."""
    p = path.lower().replace("-", "_")
    if "translate" in p:
        return "llm_translate"
    if "generate" in p or "ideas" in p:
        return "llm_content"
    return "llm_content"  # default for unknown LLM endpoints


async def _handle_llm_content(body: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Handler for generate-ideas, generate, and similar LLM content endpoints."""
    config = config or {}
    topic = (body.get("topic") or body.get("text") or "").strip() or str(body)[:8000]
    count = max(1, min(10, body.get("count") or 5))
    default_type = (config.get("gen_type") or "general").lower().strip().replace("-", "_").replace(" ", "_")
    raw_type = (body.get("gen_type") or default_type).lower().strip().replace("-", "_").replace(" ", "_")
    if raw_type in ("summarization",):
        raw_type = "summarize"
    gen_type = raw_type if raw_type in GEN_PROMPTS else default_type if default_type in GEN_PROMPTS else "general"

    from llm_helper import get_llm_for_user
    from langchain_core.messages import HumanMessage, SystemMessage

    temp = 0.3 if gen_type in ("summarize", "translate") else (0.8 if gen_type == "ideas" else 0.7)
    llm = get_llm_for_user(user_email=None, temperature=temp)
    prompt_template = GEN_PROMPTS.get(gen_type, GEN_PROMPTS["general"])
    custom_prompt = config.get("llmPrompt") or config.get("llm_prompt")
    system_content = custom_prompt if custom_prompt and gen_type == "general" else prompt_template.format(count=count)
    messages = [SystemMessage(content=system_content), HumanMessage(content=topic)]
    response = await llm.ainvoke(messages)
    text = (response.content or "").strip()

    if gen_type == "linkedin_post":
        lines = text.split("\n")
        hashtag_line = next((l.strip() for l in lines if l.strip().startswith("#")), None)
        post_lines = [l for l in lines if not l.strip().startswith("#")] if hashtag_line else lines
        content = ["\n".join(post_lines).strip()] if post_lines else [text]
    elif gen_type in ("translate", "summarize", "summarization"):
        content = [text] if text else ["No output generated."]
    elif gen_type in ("travel", "general", "linkedin_post"):
        # Long-form markdown — keep full text; UI renders raw_text
        content = [text] if text else ["No content generated."]
    else:
        ideas = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and (". " in line or ") " in line):
                line = line.split(". ", 1)[-1] if ". " in line else line.split(") ", 1)[-1]
            if line:
                ideas.append(line)
        content = ideas[:count] if ideas else [text] if text else ["No content generated."]

    return {"topic": topic, "gen_type": gen_type, "content": content, "raw_text": text}


async def _handle_llm_translate(body: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for /translate endpoint (language-translator template)."""
    text = (body.get("text") or body.get("topic") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    source_lang = body.get("source_lang") or "auto"
    target_lang = body.get("target_lang") or "en"

    from llm_helper import get_llm_for_user
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = f"You are a translator. Translate the following text from {source_lang} to {target_lang}. Return only the translation, no explanations."
    llm = get_llm_for_user(user_email=None, temperature=0.3)
    response = await llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=text)])
    translated = (response.content or "").strip() or text

    return {
        "original_text": text,
        "translated_text": translated,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }


async def _dispatch_custom_endpoint(project_id: str, action: str, body: Dict[str, Any], config: Dict[str, Any]) -> JSONResponse:
    """Dispatch to the appropriate handler for custom (non-CRUD) endpoints."""
    custom = config.get("custom_endpoints") or {}
    handler = custom.get(action) if isinstance(custom, dict) else None
    if not handler:
        raise HTTPException(status_code=404, detail=f"Custom endpoint '{action}' not found")
    try:
        if handler == "llm_content":
            result = await _handle_llm_content(body, config)
        elif handler == "llm_translate":
            result = await _handle_llm_translate(body)
        else:
            result = await _handle_llm_content(body, config)  # fallback
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"LLM not available: {e}")
    except Exception as e:
        logger.exception("[dynamic_app] custom endpoint %s error: %s", action, e)
        raise HTTPException(status_code=500, detail=str(e))


# --- CRUD routes ---


@router.get("/{project_id}/{collection}")
async def list_collection(
    project_id: str,
    collection: str,
    current: CurrentUser = Depends(resolve_user),
):
    """List all items in collection."""
    assert_project_access(project_id, current)
    config = get_project_config(project_id)
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")
    tables = config.get("tables") or []
    if collection not in tables:
        raise HTTPException(status_code=404, detail=f"Collection {collection} not found")
    conn = _get_connection(project_id, config)
    try:
        cursor = conn.execute(f'SELECT * FROM "{collection}"')
        rows = cursor.fetchall()
        return JSONResponse(content=[_row_to_dict(r) for r in rows])
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.get("/{project_id}/{collection}/{item_id}")
async def get_item(
    project_id: str,
    collection: str,
    item_id: int,
    current: CurrentUser = Depends(resolve_user),
):
    """Get one item by id."""
    assert_project_access(project_id, current)
    config = get_project_config(project_id)
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")
    tables = config.get("tables") or []
    if collection not in tables:
        raise HTTPException(status_code=404, detail=f"Collection {collection} not found")
    conn = _get_connection(project_id, config)
    try:
        cursor = conn.execute(f'SELECT * FROM "{collection}" WHERE id = ?', (item_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        return JSONResponse(content=_row_to_dict(row))
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.post("/{project_id}/{collection}")
async def create_item(
    project_id: str,
    collection: str,
    request: Request,
    current: CurrentUser = Depends(resolve_user),
):
    """Create item in collection, or dispatch to custom endpoint (generate-ideas, translate, etc.)."""
    assert_project_access(project_id, current)
    config = get_project_config(project_id)
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")
    tables = config.get("tables") or []
    custom_endpoints = config.get("custom_endpoints") or {}
    # Backward compat: legacy config may have list ["generate-ideas"] -> treat as dict
    if isinstance(custom_endpoints, list):
        custom_endpoints = {p: "llm_content" for p in custom_endpoints if isinstance(p, str)}
    if not isinstance(custom_endpoints, dict):
        custom_endpoints = {}
    # Fallback: known action paths (old projects or config missing custom_endpoints)
    KNOWN_ACTIONS = {"generate-ideas": "llm_content", "generate": "llm_content", "translate": "llm_translate"}
    if collection in KNOWN_ACTIONS and collection not in custom_endpoints:
        custom_endpoints = {**custom_endpoints, collection: KNOWN_ACTIONS[collection]}

    # Custom endpoint (generate-ideas, translate, etc.) - dispatch before CRUD
    if isinstance(custom_endpoints, dict) and collection in custom_endpoints:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        return await _dispatch_custom_endpoint(project_id, collection, body, config)

    if collection not in tables:
        raise HTTPException(status_code=404, detail=f"Collection {collection} not found")
    body = await request.json()
    extra = {collection: [k for k in body.keys() if k != "id"]}
    conn = _get_connection(project_id, config, extra_cols=extra)
    try:
        cols = [k for k in body.keys() if k != "id"]
        vals = [body[k] for k in cols]
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(f'"{c}"' for c in cols)
        conn.execute(
            f'INSERT INTO "{collection}" ({col_str}) VALUES ({placeholders})',
            vals,
        )
        conn.commit()
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        cursor = conn.execute(f'SELECT * FROM "{collection}" WHERE id = ?', (rid,))
        row = cursor.fetchone()
        return JSONResponse(content=_row_to_dict(row), status_code=201)
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.put("/{project_id}/{collection}/{item_id}")
async def update_item(
    project_id: str,
    collection: str,
    item_id: int,
    request: Request,
    current: CurrentUser = Depends(resolve_user),
):
    """Update item by id."""
    assert_project_access(project_id, current)
    config = get_project_config(project_id)
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")
    tables = config.get("tables") or []
    if collection not in tables:
        raise HTTPException(status_code=404, detail=f"Collection {collection} not found")
    body = await request.json()
    cols = [k for k in body.keys() if k != "id"]
    if not cols:
        raise HTTPException(status_code=400, detail="No fields to update")
    extra = {collection: cols}
    conn = _get_connection(project_id, config, extra_cols=extra)
    try:
        set_clause = ",".join(f'"{c}" = ?' for c in cols)
        vals = [body[k] for k in cols] + [item_id]
        conn.execute(
            f'UPDATE "{collection}" SET {set_clause} WHERE id = ?',
            vals,
        )
        conn.commit()
        cursor = conn.execute(f'SELECT * FROM "{collection}" WHERE id = ?', (item_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        return JSONResponse(content=_row_to_dict(row))
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.patch("/{project_id}/{collection}/{item_id}")
async def patch_item(
    project_id: str,
    collection: str,
    item_id: int,
    request: Request,
    current: CurrentUser = Depends(resolve_user),
):
    """Partial update (alias for PUT — generated apps may use PATCH)."""
    return await update_item(project_id, collection, item_id, request, current)


@router.delete("/{project_id}/{collection}/{item_id}")
async def delete_item(
    project_id: str,
    collection: str,
    item_id: int,
    current: CurrentUser = Depends(resolve_user),
):
    """Delete item by id."""
    assert_project_access(project_id, current)
    config = get_project_config(project_id)
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")
    tables = config.get("tables") or []
    if collection not in tables:
        raise HTTPException(status_code=404, detail=f"Collection {collection} not found")
    conn = _get_connection(project_id, config)
    try:
        conn.execute(f'DELETE FROM "{collection}" WHERE id = ?', (item_id,))
        conn.commit()
        return JSONResponse(content={"status": "deleted"})
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- LLM proxy ---


class LLMRequest(BaseModel):
    messages: List[Dict[str, str]]
    stream: bool = False


@router.post("/{project_id}/llm")
async def llm_proxy(
    project_id: str,
    request: LLMRequest,
    current: CurrentUser = Depends(resolve_user),
):
    """Proxy LLM calls with project-specific system prompt."""
    assert_project_access(project_id, current)
    config = get_project_config(project_id)
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")
    system_prompt = config.get("llmPrompt") or "You are a helpful assistant."
    try:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0.7)
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=system_prompt)]
        for m in request.messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                messages.insert(0, SystemMessage(content=content))
            elif role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=content))
        if request.stream:
            chunks = []
            async for chunk in llm.astream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    chunks.append(chunk.content)
            return JSONResponse(content={"content": "".join(chunks)})
        response = await llm.ainvoke(messages)
        return JSONResponse(content={"content": response.content})
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"LLM not available: {e}")
    except Exception as e:
        logger.exception("[dynamic_app] LLM error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
