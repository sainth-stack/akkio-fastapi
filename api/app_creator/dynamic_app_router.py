"""
Dynamic CRUD and LLM proxy for generated apps.
Single backend serves all projects via config-driven routes.
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel

from api.auth.dependencies import CurrentUser
from api.auth.request_auth import resolve_user_for_generated_app
from api.app_creator.project_access import assert_project_access

from app_builder.services.project_config import get_project_config
from app_builder.services.runtime_paths import get_projects_dir, resolve_project_root
from app_builder.services.app_data_rag import load_project_app_definition
from app_builder.services.demo_api_data import (
    anomaly_dashboard_payload,
    anomaly_reports_summary,
    default_dashboard_for_project,
    resolve_special_collection,
    resolve_special_subresource,
    supply_chain_dashboard_kpis,
)
from db import dynamic_app_store as dyn_store

logger = logging.getLogger("app_builder")

router = APIRouter(prefix="/api/apps", tags=["Dynamic Apps"])


async def _safe_json_body(request: Request) -> Dict[str, Any]:
    """Parse JSON body without raising uncaught decode errors (→ 500 HTML)."""
    if not request.headers.get("content-type", "").startswith("application/json"):
        return {}
    try:
        raw = await request.body()
        if not raw or not raw.strip():
            return {}
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {"data": data}
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON body")


def _format_llm_prompt(template: str, count: int) -> str:
    try:
        return template.format(count=count)
    except (KeyError, ValueError):
        return template


def _ensure_collection_registered(config: Dict[str, Any], collection: str) -> Dict[str, Any]:
    """If frontend hits a collection missing from config.tables, register it from fields."""
    out = dict(config)
    tables = list(out.get("tables") or [])
    fields = dict(out.get("fields") or {})
    if collection in tables:
        return out
    if collection in fields:
        tables.append(collection)
        out["tables"] = tables
        return out
    out["tables"] = tables + [collection]
    out.setdefault("fields", {})[collection] = ["title"]
    return out


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
    if custom_prompt and gen_type == "general":
        system_content = custom_prompt
    else:
        system_content = _format_llm_prompt(prompt_template, count)
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
        raise HTTPException(status_code=503, detail=f"AI service is not configured: {e}")
    except Exception as e:
        logger.exception("[dynamic_app] custom endpoint %s error: %s", action, e)
        raise HTTPException(
            status_code=503,
            detail="AI request failed. Check API keys in server .env and try again.",
        ) from e


# --- App assistant (must register before /{collection} CRUD or "chat" is treated as a table) ---

_RESERVED_PATH_SEGMENTS = frozenset({"chat", "llm", "predict", "observability"})


class LLMRequest(BaseModel):
    messages: List[Dict[str, str]]
    stream: bool = False


class AppChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    page_id: Optional[str] = None
    app_definition: Optional[Dict[str, Any]] = None


def _client_supplied_app_definition(body: Optional[AppChatRequest]) -> bool:
    """Browser embeds data.json in the chat request — no server project copy required."""
    if not body or not isinstance(body.app_definition, dict):
        return False
    d = body.app_definition
    if d.get("appKind") == "ecommerce" and isinstance(d.get("products"), list) and d.get("products"):
        return True
    data = d.get("data") or d.get("mock")
    if isinstance(data, dict) and data:
        return True
    pages = d.get("pages")
    return isinstance(pages, list) and len(pages) > 0


async def _optional_project_access(
    project_id: str,
    request: Request,
    body: Optional[AppChatRequest] = None,
) -> None:
    """JWT when provided; else allow chat with inline app_definition or on-disk project data."""
    if _client_supplied_app_definition(body):
        return

    from fastapi.security import HTTPAuthorizationCredentials

    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=auth.split(" ", 1)[1].strip())
        current = await resolve_user_for_generated_app(request, creds)
        assert_project_access(project_id, current)
        return
    token = (request.query_params.get("access_token") or "").strip()
    if token:
        preview_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        current = await resolve_user_for_generated_app(request, preview_creds)
        assert_project_access(project_id, current)
        return
    from app_builder.services.app_data_rag import load_project_app_definition

    if load_project_app_definition(project_id):
        return
    raise HTTPException(
        status_code=401,
        detail="Chat requires sign-in, preview access token, or app data in the request.",
    )


@router.post("/{project_id}/chat")
async def app_data_chat(
    project_id: str,
    body: AppChatRequest,
    request: Request,
):
    """RAG chat over the generated app's KPIs, charts, and tables."""
    await _optional_project_access(project_id, request, body)
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages required")
    try:
        from app_builder.services.app_chat_service import answer_app_chat

        result = await answer_app_chat(
            project_id,
            body.messages,
            page_id=body.page_id,
            app_definition=body.app_definition,
        )
        content = result.get("content") if isinstance(result, dict) else str(result)
        usage = result.get("observability") if isinstance(result, dict) else None
        if usage:
            from app_builder.services.llm_observability_store import append_observability_event

            append_observability_event(project_id, usage)
        payload: Dict[str, Any] = {"role": "assistant", "content": content}
        if usage:
            payload["usage"] = usage
        return JSONResponse(content=payload)
    except ValueError as e:
        msg = str(e)
        low = msg.lower()
        if "api key" in low or ("llm" in low and "configured" in low):
            raise HTTPException(status_code=503, detail=msg) from e
        raise HTTPException(status_code=404, detail=msg) from e
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"AI service is not configured: {e}") from e
    except Exception as e:
        logger.exception("[dynamic_app] app chat error: %s", e)
        err = str(e)
        if "api key" in err.lower():
            raise HTTPException(status_code=503, detail=err) from e
        raise HTTPException(
            status_code=503,
            detail="Chat failed. Check server logs and LLM API keys in akkio-fastapi/.env.",
        ) from e


class PredictRequest(BaseModel):
    features: Dict[str, Any] = {}
    prompt: Optional[str] = None
    target_field: Optional[str] = None
    refresh_baseline_only: bool = False
    app_definition: Optional[Dict[str, Any]] = None


@router.post("/{project_id}/chat/rag-export")
async def rag_export(
    project_id: str,
    body: AppChatRequest,
    request: Request,
):
    """Return full RAG context for PDF download in the chat UI."""
    await _optional_project_access(project_id, request, body)
    from app_builder.services.app_data_rag import build_rag_context, load_project_app_definition

    defn = body.app_definition or load_project_app_definition(project_id)
    if not defn:
        raise HTTPException(status_code=404, detail="App definition not found")
    meta = defn.get("meta") or {}
    query = "full export of KPIs charts tables products and dashboard data"
    context = build_rag_context(defn, query, focus_page_id=body.page_id)
    return JSONResponse(
        content={
            "appName": meta.get("appName") or "Application",
            "context": context,
            "exportedAt": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }
    )


@router.post("/{project_id}/predict")
async def predict_synthetic(
    project_id: str,
    body: PredictRequest,
    request: Request,
):
    """Synthetic forecast from operational inputs (manual Predict page or chat)."""
    pseudo = AppChatRequest(messages=[], app_definition=body.app_definition)
    await _optional_project_access(project_id, request, pseudo)
    from app_builder.services.synthetic_predict import (
        _collect_kpi_bag,
        default_features_from_definition,
        run_live_prediction,
    )
    from app_builder.services.app_data_rag import load_project_app_definition

    defn = body.app_definition or load_project_app_definition(project_id)
    defn_dict = defn if isinstance(defn, dict) else None
    if defn_dict:
        from app_builder.services.predict_config import attach_predict_config

        defn_dict = attach_predict_config(dict(defn_dict))
    if body.refresh_baseline_only:
        cfg = (defn_dict or {}).get("meta", {}).get("predictConfig") if defn_dict else None
        baseline = default_features_from_definition(defn_dict)
        payload: Dict[str, Any] = {
            "baseline": baseline,
            "kpi_snapshot": _collect_kpi_bag(defn_dict),
        }
        if cfg:
            payload["predictConfig"] = cfg
        return JSONResponse(content=payload)
    result = run_live_prediction(
        defn_dict,
        features=body.features or {},
        prompt=body.prompt or "",
        target_field=body.target_field,
    )
    return JSONResponse(content=result)


@router.get("/{project_id}/observability")
async def get_llm_observability(
    project_id: str,
    request: Request,
    limit: int = 100,
):
    """LLM call log: prompts, outputs, tokens, and estimated USD cost."""
    await _optional_project_access(project_id, request, None)
    from app_builder.services.llm_observability_store import (
        list_observability_events,
        summarize_observability,
    )

    lim = max(1, min(int(limit or 100), 500))
    events = list_observability_events(project_id, limit=lim)
    summary = summarize_observability(events)
    return JSONResponse(content={"events": events, "summary": summary})


@router.get("/{project_id}/definition")
async def get_app_definition(
    project_id: str,
    request: Request,
):
    """Latest data.json for hosted preview (after upload or rebuild)."""
    await _optional_project_access(project_id, request, None)
    from app_builder.services.data_import_service import load_data_json_from_project
    from app_builder.services.runtime_paths import resolve_project_root

    defn = load_project_app_definition(project_id)
    if not defn:
        root = resolve_project_root(project_id)
        defn = load_data_json_from_project(root)
    if not defn:
        raise HTTPException(status_code=404, detail="App definition not found")
    from app_builder.services.json_app_sample_export import load_builder_prompt_context
    from app_builder.services.operational_metrics_compute import apply_source_data_pipeline

    requirement, prd = load_builder_prompt_context(project_id)
    defn = apply_source_data_pipeline(
        defn,
        requirement=requirement,
        prd=prd,
        seed_if_empty=True,
    )
    return JSONResponse(content=defn)


@router.get("/{project_id}/data/template.xlsx")
async def download_data_import_template(
    project_id: str,
    request: Request,
):
    """Excel template (one sheet per dataKey) for edit-and-upload workflow."""
    await _optional_project_access(project_id, request, None)
    from app_builder.services.sample_data_export import build_sample_data_xlsx

    try:
        xlsx_path = build_sample_data_xlsx(project_id)
    except Exception as e:
        logger.exception("[dynamic_app] template export failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Could not build data template: {e}") from e

    def _cleanup(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    app_name = project_id
    defn = load_project_app_definition(project_id)
    if isinstance(defn, dict):
        app_name = str((defn.get("meta") or {}).get("appName") or project_id)
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", app_name)[:48]

    return FileResponse(
        path=xlsx_path,
        filename=f"{safe}_data_template.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        background=BackgroundTask(_cleanup, xlsx_path),
    )


@router.post("/{project_id}/data/import")
async def import_app_data(
    project_id: str,
    request: Request,
    file: UploadFile = File(...),
    target_key: Optional[str] = Form(None),
):
    """Upload CSV, JSON, or Excel and merge into frontend/src/mock/data.json datasets."""
    await _optional_project_access(project_id, request, AppChatRequest(messages=[]))

    from app_builder.services.data_import_service import (
        list_import_targets,
        load_data_json_from_project,
        merge_upload_bytes,
        save_data_json_to_project,
    )
    from app_builder.services.runtime_paths import resolve_project_root

    root = resolve_project_root(project_id)
    defn = load_data_json_from_project(root) or load_project_app_definition(project_id)
    if not defn:
        raise HTTPException(status_code=404, detail="App has no data.json — build the app first.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        updated, summary = merge_upload_bytes(
            defn,
            filename=file.filename or "upload.csv",
            raw=raw,
            target_key=target_key or None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    save_data_json_to_project(project_id, updated)
    return JSONResponse(
        content={
            "message": "Data imported",
            "updatedKeys": summary.get("updatedKeys") or [],
            "summary": summary,
            "targets": list_import_targets(updated),
        }
    )


@router.post("/{project_id}/llm")
async def llm_proxy(
    project_id: str,
    request: LLMRequest,
    current: CurrentUser = Depends(resolve_user_for_generated_app),
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
        content = str(getattr(response, "content", response) or "")
        from llm_config import get_llm_config

        llm_cfg = get_llm_config(None)
        model = str(llm_cfg.get("model") or "gpt-4o-mini")
        last_user = ""
        for m in reversed(request.messages):
            if (m.get("role") or "").lower() == "user":
                last_user = str(m.get("content") or "")
                break
        from app_builder.services.llm_observability_store import (
            append_observability_event,
            build_observability_event,
        )

        usage = build_observability_event(
            source="llm_proxy",
            model=model,
            prompt=last_user,
            output=content,
            llm_response_obj=response,
        )
        append_observability_event(project_id, usage)
        return JSONResponse(content={"content": content, "usage": usage})
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"AI service is not configured: {e}")
    except Exception as e:
        logger.exception("[dynamic_app] LLM error: %s", e)
        raise HTTPException(
            status_code=503,
            detail="AI request failed. Check API keys in server .env and try again.",
        ) from e


# --- CRUD routes ---


@router.get("/{project_id}/{collection}")
async def list_collection(
    project_id: str,
    collection: str,
    current: CurrentUser = Depends(resolve_user_for_generated_app),
):
    """List all items in collection."""
    assert_project_access(project_id, current)
    seed = _load_seed_data(project_id)
    special = resolve_special_collection(collection, seed, project_id=project_id)
    if special is not None:
        return JSONResponse(content=special)

    config = get_project_config(project_id)
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")
    config = _ensure_collection_registered(config, collection)
    tables = config.get("tables") or []
    if collection not in tables:
        raise HTTPException(status_code=404, detail=f"Collection {collection} not found")
    try:
        items = dyn_store.list_items(project_id, collection)
        return JSONResponse(content=items)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database error: {e}") from e


_DEMO_KPIS: Dict[str, Any] = {
    "total_lots": 142,
    "pending_inspections": 23,
    "released": 98,
    "held": 12,
    "rejected": 9,
    "open_capa": 7,
    "incoming_lots": 18,
    "total_items": 3420,
    "low_stock_alerts": 14,
    "total_orders": 287,
    "pending_orders": 34,
    "total_suppliers": 67,
}

_KPI_SUBPATHS = frozenset({"kpis", "stats", "summary", "overview", "metrics", "counters"})


def _project_has_anomaly_ui(project_id: str) -> bool:
    try:
        root = resolve_project_root(project_id)
        return os.path.exists(os.path.join(root, "frontend/src/pages/AnomalyFeedPage.tsx"))
    except Exception:
        return False


def _load_seed_data(project_id: str) -> Dict[str, Any]:
    """Load backend/seed_data.json for a project if it exists; returns empty dict otherwise."""
    try:
        project_root = resolve_project_root(project_id)
        seed_file = os.path.join(project_root, "backend", "seed_data.json")
        if os.path.exists(seed_file):
            with open(seed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("[dynamic_app] could not load seed_data.json for %s: %s", project_id, e)
    return {}


@router.get("/{project_id}/{collection}/{item_id}")
async def get_item(
    project_id: str,
    collection: str,
    item_id: str,
    request: Request,
    current: CurrentUser = Depends(resolve_user_for_generated_app),
):
    """Get one item by id, or handle named sub-resource endpoints (e.g. dashboard/kpis)."""
    assert_project_access(project_id, current)
    config = get_project_config(project_id)
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")

    # ── Sub-resource path (e.g. /dashboard/kpis, /reports/summary) ────────────
    if not item_id.lstrip("-").isdigit():
        subpath = item_id.lower()
        compound_key = f"{collection}/{subpath}"

        # 1. Try seed_data.json + domain-aware demo payloads
        seed = _load_seed_data(project_id)
        if collection == "reports" and subpath == "template":
            from app_builder.services.sample_report_templates import resolve_reports_template_payload

            template_id = request.query_params.get("template_id")
            return JSONResponse(
                content=resolve_reports_template_payload(
                    seed, template_id=template_id, project_id=project_id
                )
            )
        sub_payload = resolve_special_subresource(collection, subpath, seed)
        if sub_payload is not None:
            return JSONResponse(content=sub_payload)
        if compound_key in seed:
            return JSONResponse(content=seed[compound_key])
        if subpath in seed:
            return JSONResponse(content=seed[subpath])

        # 2. KPI/stats fallback — return demo data so dashboards never show zeros
        if subpath in _KPI_SUBPATHS or collection in ("dashboard", "metrics", "stats", "analytics"):
            kpi_data = (
                seed.get("kpis")
                or seed.get("dashboard/kpis")
                or (
                    default_dashboard_for_project(project_id, seed)
                    if _project_has_anomaly_ui(project_id)
                    else supply_chain_dashboard_kpis(project_id)
                )
            )
            logger.info(
                "[dynamic_app] serving demo KPIs for %s/%s/%s (seed_data absent)",
                project_id, collection, item_id,
            )
            return JSONResponse(content=kpi_data)

        # 3. Unknown sub-resource — 404
        raise HTTPException(status_code=404, detail=f"Sub-resource '{collection}/{item_id}' not found")

    # ── Integer DB lookup (original behaviour) ─────────────────────────────────
    int_id = int(item_id)
    config = _ensure_collection_registered(config, collection)
    tables = config.get("tables") or []
    if collection not in tables:
        raise HTTPException(status_code=404, detail=f"Collection {collection} not found")
    try:
        row = dyn_store.get_item(project_id, collection, int_id)
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        return JSONResponse(content=row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database error: {e}") from e


@router.post("/{project_id}/{collection}")
async def create_item(
    project_id: str,
    collection: str,
    request: Request,
    current: CurrentUser = Depends(resolve_user_for_generated_app),
):
    """Create item in collection, or dispatch to custom endpoint (generate-ideas, translate, etc.)."""
    if collection in _RESERVED_PATH_SEGMENTS:
        raise HTTPException(status_code=404, detail=f"Unknown collection: {collection}")
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
        body = await _safe_json_body(request)
        return await _dispatch_custom_endpoint(project_id, collection, body, config)

    config = _ensure_collection_registered(config, collection)
    tables = config.get("tables") or []
    if collection not in tables:
        raise HTTPException(status_code=404, detail=f"Collection {collection} not found")
    body = await _safe_json_body(request)
    try:
        row = dyn_store.create_item(project_id, collection, body)
        return JSONResponse(content=row, status_code=201)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database error: {e}") from e


@router.put("/{project_id}/{collection}/{item_id}")
async def update_item(
    project_id: str,
    collection: str,
    item_id: str,
    request: Request,
    current: CurrentUser = Depends(resolve_user_for_generated_app),
):
    """Update item by id."""
    assert_project_access(project_id, current)
    if not item_id.lstrip("-").isdigit():
        raise HTTPException(status_code=400, detail=f"Invalid item id: {item_id}")
    int_id = int(item_id)
    config = get_project_config(project_id)
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")
    tables = config.get("tables") or []
    config = _ensure_collection_registered(config, collection)
    tables = config.get("tables") or []
    if collection not in tables:
        raise HTTPException(status_code=404, detail=f"Collection {collection} not found")
    body = await _safe_json_body(request)
    cols = [k for k in body.keys() if k != "id"]
    if not cols:
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        row = dyn_store.update_item(project_id, collection, int_id, body)
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        return JSONResponse(content=row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database error: {e}") from e


@router.patch("/{project_id}/{collection}/{item_id}")
async def patch_item(
    project_id: str,
    collection: str,
    item_id: str,
    request: Request,
    current: CurrentUser = Depends(resolve_user_for_generated_app),
):
    """Partial update (alias for PUT — generated apps may use PATCH)."""
    return await update_item(project_id, collection, item_id, request, current)


@router.delete("/{project_id}/{collection}/{item_id}")
async def delete_item(
    project_id: str,
    collection: str,
    item_id: str,
    current: CurrentUser = Depends(resolve_user_for_generated_app),
):
    """Delete item by id."""
    assert_project_access(project_id, current)
    if not item_id.lstrip("-").isdigit():
        raise HTTPException(status_code=400, detail=f"Invalid item id: {item_id}")
    int_id = int(item_id)
    config = get_project_config(project_id)
    if not config:
        raise HTTPException(status_code=404, detail="Project not found")
    tables = config.get("tables") or []
    config = _ensure_collection_registered(config, collection)
    tables = config.get("tables") or []
    if collection not in tables:
        raise HTTPException(status_code=404, detail=f"Collection {collection} not found")
    try:
        if not dyn_store.delete_item(project_id, collection, int_id):
            raise HTTPException(status_code=404, detail="Not found")
        return JSONResponse(content={"status": "deleted"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database error: {e}") from e

