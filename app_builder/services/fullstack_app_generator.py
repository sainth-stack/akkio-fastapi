"""Production fullstack app generator — complete MUI + FastAPI MVP from PRD/architecture."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from app_builder.services.fullstack_codegen import get_fullstack_scaffold_files
from app_builder.services.fullstack_frontend_generator import frontend_files, theme_from_tokens
from app_builder.services.fullstack_backend_generator import backend_files
from app_builder.services.fullstack_docchat_generator import (
    doc_chat_backend_files,
    doc_chat_frontend_files,
    is_doc_chat_domain,
)
from app_builder.services.fullstack_ecommerce_generator import (
    ecommerce_frontend_files,
    ecommerce_backend_files,
    is_ecommerce_domain,
)
from app_builder.services.fullstack_anomaly_generator import (
    anomaly_detection_frontend_files,
    anomaly_detection_backend_files,
    is_anomaly_detection_domain,
)

logger = logging.getLogger("app_builder")

# ---------------------------------------------------------------------------
# Track routing
# ---------------------------------------------------------------------------
# Domains in STATIC_DOMAINS use the existing fullstack (React + FastAPI + SQLite)
# generators and MUST NOT be changed.  Every other domain is routed to the new
# Lovable-style frontend-only track (React + mock data, no backend).
STATIC_DOMAINS: frozenset = frozenset({"ecommerce", "doc_chat", "anomaly_detection"})


def resolve_track(domain: str) -> str:
    """Return the generation track for a classified domain.

    Returns:
        "legacy"         – existing fullstack backend+frontend pipeline
        "frontend_only"  – new Lovable-style SPA with mock data, no backend
    """
    return "legacy" if domain in STATIC_DOMAINS else "frontend_only"

# ──────────────────────────────────────────────────────────────────────────────
# LLM App-type Classifier
# ──────────────────────────────────────────────────────────────────────────────

APP_TYPE_CLASSIFIER_PROMPT = """Classify the following app requirement into exactly one app type.

Requirement:
{requirement}

PRD summary (first 500 chars):
{prd_summary}

Return EXACTLY one label from this list (nothing else):
ecommerce | doc_chat | anomaly_detection | analytics_dashboard | crm | inventory | hr_onboarding | finance | quality | generic

Classification rules:
- ecommerce: shopping cart, products, checkout, orders, add to cart, online store
- doc_chat: document upload, PDF chat, knowledge base Q&A, chat over documents
- anomaly_detection: anomaly detection, outlier detection, metric monitoring, threshold alerts, z-score, time-series anomaly
- analytics_dashboard: business intelligence, BI dashboard, KPI reporting, data analytics, visualization
- crm: CRM, customers, leads, sales pipeline, customer service, contacts
- inventory: inventory management, stock levels, warehouse, supply chain, SKU
- hr_onboarding: HR management, employee onboarding, payroll, leave management, workforce
- finance: financial management, accounting, invoices, expenses, budgets, cash flow
- quality: supplier quality, incoming material, CAPA, defect rate, inspection, QMS
- generic: anything that doesn't clearly fit the above categories

Return ONLY the single label with no punctuation, no explanation."""


async def _classify_with_llm(requirement: str, prd: str = "") -> Optional[str]:
    """Use a fast LLM call to classify the app type. Returns None on any failure."""
    try:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(None, temperature=0.1)
    except Exception:
        return None
    prompt = APP_TYPE_CLASSIFIER_PROMPT.format(
        requirement=(requirement or "")[:1000],
        prd_summary=(prd or "")[:500],
    )
    try:
        resp = await llm.ainvoke(prompt)
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip().lower()
        label = text.split()[0].rstrip(".,;:") if text.split() else ""
        valid = {
            "ecommerce", "doc_chat", "anomaly_detection", "analytics_dashboard",
            "crm", "inventory", "hr_onboarding", "finance", "quality", "generic",
        }
        if label in valid:
            logger.info("[classify] LLM classified app as: %s", label)
            return label
        logger.debug("[classify] LLM returned unrecognised label %r — ignoring", label)
    except Exception as exc:
        logger.debug("[classify] LLM classification failed: %s", exc)
    return None


def _keyword_classify(requirement: str, prd: str = "", uiux: str = "") -> str:
    """Pure keyword-based app-type classification (fast, deterministic fallback)."""
    if is_doc_chat_domain(requirement, prd, uiux):
        return "doc_chat"
    if is_ecommerce_domain(requirement, prd, uiux):
        return "ecommerce"
    if is_anomaly_detection_domain(requirement, prd, uiux):
        return "anomaly_detection"
    if is_quality_domain(requirement, prd, uiux):
        return "quality"
    return "generic"


def resolve_app_mode(requirement: str, prd: str = "", uiux: str = "") -> str:
    """Classify app type using LLM (with keyword fallback).

    Runs the async LLM classifier in an isolated thread so this sync function
    can be called from any context (event loop or plain thread).
    Falls back to keyword matching on any error or timeout.
    """
    try:
        import asyncio
        import concurrent.futures

        def _run_async() -> Optional[str]:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_classify_with_llm(requirement, prd))
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_async)
            result = future.result(timeout=8)
            if result:
                return result
    except Exception as exc:
        logger.debug("[resolve_app_mode] LLM classify unavailable (%s) — using keywords", exc)

    return _keyword_classify(requirement, prd, uiux)


def is_quality_domain(requirement: str, prd: str = "", uiux: str = "") -> bool:
    text = "\n".join([requirement or "", prd or "", uiux or ""]).lower()
    # Strong single-phrase matches → quality
    strong = (
        "supplier quality management", "incoming material release",
        "supplier quality & incoming", "quality management system",
        "incoming lot inspection",
    )
    if any(k in text for k in strong):
        return True
    # Multiple supporting keywords → quality
    keys = ("supplier quality", "incoming lot", "incoming material", "capa", "ppm", "release decision",
            "defect rate", "inspection result", "quality control", "quality assurance", "supplier scorecard")
    return sum(1 for k in keys if k in text) >= 2


def extract_app_title(requirement: str, prd: str = "") -> str:
    quoted = re.search(r'"([^"]{6,90})"', requirement or "")
    if quoted:
        return quoted.group(1).strip()
    heading = re.search(r"^#\s+(.+)$", prd or "", re.M)
    if heading and "product" not in heading.group(1).lower():
        return heading.group(1).strip()[:90]
    first = (requirement or "Enterprise Application").strip().split("\n")[0].strip()
    if len(first) < 28 or re.match(r"^(create|add|build|make|update|two screen|2 screen)", first, re.I):
        if "supplier quality" in (prd or "").lower() or "incoming material" in (prd or "").lower():
            return "Supplier Quality & Incoming Material Release"
        if is_doc_chat_domain(requirement, prd):
            return "Document Upload & Chat"
        if is_ecommerce_domain(requirement, prd):
            return "E-Commerce Shopping Cart"
        if is_anomaly_detection_domain(requirement, prd):
            return "Anomaly Detection Platform"
        return "Enterprise Application"
    return first[:90] if first else "Enterprise Application"


def generate_fullstack_application(
    requirement: str,
    prd: str = "",
    uiux: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    files = dict(get_fullstack_scaffold_files())
    title = extract_app_title(requirement, prd)
    colors = theme_from_tokens(design_tokens, uiux=uiux)
    mode = resolve_app_mode(requirement, prd, uiux)

    # ── New frontend-only track ────────────────────────────────────────────
    # All domains NOT in STATIC_DOMAINS are handled by the Lovable-style
    # frontend-only pipeline (no FastAPI backend, no SQLite, mock data only).
    if mode not in STATIC_DOMAINS:
        logger.info("[generator] domain=%s → frontend_only track", mode)
        try:
            from app_builder.services.frontend_only_pipeline import generate_frontend_only_app
            fo_files = generate_frontend_only_app(
                requirement=requirement,
                prd=prd,
                uiux=uiux,
                architecture=architecture,
                design_tokens=design_tokens,
                domain=mode,
            )
            if fo_files:
                return fo_files
        except Exception as _fe_exc:
            logger.warning(
                "[generator] frontend_only pipeline failed (%s) — falling back to AI generator",
                _fe_exc,
            )
        # Fallback: AI-driven generation (same as the existing inventory/crm/generic path)
        enriched = (
            f"[{mode.replace('_', ' ').title()} Application] {requirement}"
            if mode not in ("generic", "")
            else requirement
        )
        ai_files = _run_ai_generator(enriched, prd, uiux, architecture, design_tokens)
        if ai_files:
            files.update(ai_files)
        else:
            files.update(frontend_files(title, colors, False))
            files.update(backend_files(title, False))
        files.pop("frontend/src/App.jsx", None)
        files.pop("frontend/src/styles/app.css", None)
        return files
    # ── Legacy static-domain track ─────────────────────────────────────────
    if mode == "doc_chat":
        files.update(doc_chat_frontend_files(title, colors))
        files.update(doc_chat_backend_files(title))
        files["backend/requirements.txt"] = _doc_chat_requirements()
        for stale in (
            "frontend/src/pages/DashboardPage.tsx",
            "frontend/src/pages/ResourceListPage.tsx",
            "frontend/src/pages/SupplierDetailPage.tsx",
            "frontend/src/pages/LotDetailPage.tsx",
            "frontend/src/pages/InspectionPage.tsx",
            "frontend/src/pages/CapaPage.tsx",
            "frontend/src/pages/ReportsPage.tsx",
            "frontend/src/pages/AIAssistantPage.tsx",
            "backend/risk_engine.py",
        ):
            files.pop(stale, None)
    elif mode == "ecommerce":
        files.update(ecommerce_frontend_files(title, colors, requirement=requirement, prd=prd))
        files.update(ecommerce_backend_files(title, requirement=requirement, prd=prd))
        files["backend/requirements.txt"] = _ecommerce_requirements()
        # remove quality-only stale pages
        for stale in (
            "frontend/src/pages/ResourceListPage.tsx",
            "frontend/src/pages/SupplierDetailPage.tsx",
            "frontend/src/pages/LotDetailPage.tsx",
            "frontend/src/pages/InspectionPage.tsx",
            "frontend/src/pages/CapaPage.tsx",
            "frontend/src/pages/ReportsPage.tsx",
            "frontend/src/pages/AIAssistantPage.tsx",
            "backend/risk_engine.py",
        ):
            files.pop(stale, None)
    elif mode == "anomaly_detection":
        files.update(
            anomaly_detection_frontend_files(
                title,
                colors,
                requirement=requirement,
                prd=prd,
                uiux=uiux,
                architecture=architecture,
                design_tokens=design_tokens,
            )
        )
        files.update(
            anomaly_detection_backend_files(
                title,
                requirement=requirement,
                prd=prd,
                colors=colors,
                uiux=uiux,
                architecture=architecture,
                design_tokens=design_tokens,
            )
        )
        files["backend/requirements.txt"] = _anomaly_requirements()
        for stale in (
            # NOTE: DashboardPage.tsx is intentionally NOT removed for anomaly_detection
            # because the anomaly App.tsx imports and routes to it.
            "frontend/src/pages/ResourceListPage.tsx",
            "frontend/src/pages/SupplierDetailPage.tsx",
            "frontend/src/pages/LotDetailPage.tsx",
            "frontend/src/pages/InspectionPage.tsx",
            "frontend/src/pages/CapaPage.tsx",
            "frontend/src/pages/AIAssistantPage.tsx",
            "backend/risk_engine.py",
        ):
            files.pop(stale, None)
    # All other domains (quality, inventory, crm, analytics_dashboard, finance,
    # hr_onboarding, generic, …) are now handled by the frontend_only track ABOVE.
    # This branch is kept only as a safety net if resolve_track routing changes.
    else:
        enriched_req = (
            f"[{mode.replace('_', ' ').title()} Application] {requirement}"
            if mode not in ("generic", "")
            else requirement
        )
        ai_files = _run_ai_generator(enriched_req, prd, uiux, architecture, design_tokens)
        if ai_files:
            files.update(ai_files)
        else:
            files.update(frontend_files(title, colors, False))
            files.update(backend_files(title, False))
    files.pop("frontend/src/App.jsx", None)
    files.pop("frontend/src/styles/app.css", None)
    return files


def _has_ai_generated_pages(files: Dict[str, str]) -> bool:
    """True if files already contain real LLM-generated page content (not thin shells)."""
    page_files = [
        p for p in files
        if p.startswith("frontend/src/pages/") and p.endswith(".tsx")
        and p not in ("frontend/src/pages/LoginPage.tsx",)
    ]
    if len(page_files) < 2:
        return False
    # Real pages are substantial — if any page has real content (>500 chars) it's AI-generated
    substantive = sum(1 for p in page_files if len(files.get(p, "")) > 500)
    return substantive >= 2


def fill_missing_fullstack_files(
    files: Dict[str, str],
    requirement: str,
    prd: str = "",
    uiux: str = "",
    architecture: Optional[Dict[str, Any]] = None,
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    out = {}
    for path, content in (files or {}).items():
        if isinstance(path, str) and isinstance(content, str):
            out[path] = content

    # If the incoming files already have substantial AI-generated pages, ONLY fill
    # truly missing infrastructure files (theme, auth, mock, App.tsx) — never overwrite pages.
    # This preserves LLM-generated pharma/domain-specific content.
    if _has_ai_generated_pages(out):
        generated = generate_fullstack_application(requirement, prd, uiux, architecture, design_tokens)
        from app_builder.services.fullstack_codegen import FULLSTACK_FROZEN
        # Only fill files that are completely missing (not pages that already exist)
        infra_only = {
            "frontend/src/theme.ts", "frontend/src/auth.ts", "frontend/src/App.tsx",
            "frontend/src/layout/AppLayout.tsx", "frontend/src/api/mock.ts",
            "backend/main.py", "backend/database.py", "backend/auth.py",
        }
        for path, content in generated.items():
            if path in FULLSTACK_FROZEN:
                continue
            if path.endswith(".jsx") or path.endswith("app.css"):
                continue
            # Only inject if the file is missing OR it's an infra file that looks like a thin shell
            current = out.get(path, "")
            if not current:
                out[path] = content
            elif path in infra_only and _is_thin_shell(path, current):
                out[path] = content
        out.pop("frontend/src/App.jsx", None)
        out.pop("frontend/src/styles/app.css", None)
        return out

    # No AI pages present — run normal fill logic
    generated = generate_fullstack_application(requirement, prd, uiux, architecture, design_tokens)
    mode = resolve_app_mode(requirement, prd, uiux)

    if mode in ("doc_chat", "ecommerce", "anomaly_detection"):
        from app_builder.services.fullstack_codegen import FULLSTACK_FROZEN
        for path, content in generated.items():
            if path.endswith(".jsx") or path.endswith("app.css"):
                continue
            if path in FULLSTACK_FROZEN:
                continue
            out[path] = content
    else:
        for path, content in generated.items():
            current = out.get(path, "")
            if path.endswith(".jsx") or path.endswith("app.css"):
                continue
            if not current or _is_thin_shell(path, current):
                out[path] = content

    out.pop("frontend/src/App.jsx", None)
    out.pop("frontend/src/styles/app.css", None)
    return out


def _is_thin_shell(path: str, content: str) -> bool:
    text = content or ""
    if path.endswith("App.tsx"):
        if "UploadPage" in text or "/upload" in text:
            return "/chat" not in text
        return text.count("<Route") < 6
    if path.endswith("AppShell.tsx"):
        if "Upload documents" in text or "Document chat" in text:
            return False
        return "Quality Operations" not in text or "MuiChip" not in text
    if path.endswith("UploadPage.tsx"):
        return "CloudUploadOutlinedIcon" not in text
    if path.endswith("routes.py"):
        if "/api/documents" in text:
            return "/api/chat/ask" not in text
        return "/api/suppliers" not in text
    if path.endswith("theme.ts"):
        return "MuiListItemButton" not in text or "primary.dark" not in text
    if path.endswith("PageHeader.tsx"):
        return True
    if path.endswith("KPICard.tsx"):
        return "borderLeft" not in text
    if path.endswith("DashboardPage.tsx"):
        return "recharts" not in text.lower() and "TrendChart" not in text and "total_orders" not in text
    if path.endswith("mock.ts"):
        if "mock-jwt" not in text:
            return True
        if "supplier-quality-manual" in text or ("chunks" in text and "documents" in text):
            return False
        # Any substantial mock.ts that has mock-jwt + mockFetch/auth is domain-specific — preserve it.
        # The old "ecommerce" label was too narrow; pharma/HR/food apps also use mockFetch+auth/login.
        if "mockFetch" in text and "auth/login" in text and len(text) > 500:
            return False
        return "ppm_trend" not in text
    if path.endswith("README.md"):
        return "demo users" not in text.lower() and "capa" not in text.lower() and "cash on delivery" not in text.lower()
    if path.endswith("models.py"):
        if "class Document" in text:
            return "DocumentChunk" not in text
        if "class Product" in text:
            return "class Order" not in text
        return "class Supplier" not in text
    if path.endswith("seed.py"):
        return "LOT-20260925" not in text and "Laptop" not in text
    if path.endswith("CartPage.tsx"):
        return "CartProvider" not in text and "useCart" not in text
    if path.endswith("CheckoutPage.tsx"):
        return "Place Order" not in text and "shipping_address" not in text
    if path.endswith("OrderConfirmationPage.tsx"):
        return "CheckCircle" not in text
    if path.endswith("CartContext.tsx"):
        return "createContext" not in text
    return False


def _inventory_requirements(requirement: str) -> str:
    return f"""Supply chain and inventory management system. Features:
    - Inventory tracking with stock levels and alerts
    - Order management (purchase orders, sales orders)
    - Supplier management and vendor tracking
    - Warehouse/location management
    - Reports and analytics on inventory turnover
    Specific requirement: {requirement}"""


def _run_ai_generator(
    requirement: str,
    prd: str,
    uiux: str,
    architecture: Optional[Dict[str, Any]],
    design_tokens: Optional[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    """Run the async AI generator in an isolated event loop (sync wrapper)."""
    try:
        from app_builder.services.fullstack_ai_generator import generate_fullstack_app_with_ai
        import asyncio
        import concurrent.futures

        def _run() -> Dict[str, str]:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    generate_fullstack_app_with_ai(
                        requirement=requirement,
                        prd=prd,
                        uiux=uiux,
                        architecture=architecture or {},
                        design_tokens=design_tokens,
                    )
                )
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            return future.result(timeout=120)
    except Exception as exc:
        logger.warning("[ai-fallback] AI generator failed: %s — falling back to generic shell", exc)
        return None


def _ecommerce_requirements() -> str:
    return """fastapi
uvicorn
sqlalchemy
pydantic
pydantic-settings
psycopg2-binary
python-jose[cryptography]
passlib[bcrypt]
python-dotenv
alembic
pytest
httpx
"""


def _anomaly_requirements() -> str:
    return """fastapi
uvicorn
pydantic
python-dotenv
pytest
httpx
"""


def _doc_chat_requirements() -> str:
    return """fastapi
uvicorn
sqlalchemy
pydantic
pydantic-settings
psycopg2-binary
python-jose[cryptography]
passlib[bcrypt]
python-dotenv
alembic
python-multipart
pypdf
pytest
httpx
"""
