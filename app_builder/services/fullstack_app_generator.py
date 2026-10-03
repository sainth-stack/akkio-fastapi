"""Production fullstack app generator — complete MUI + FastAPI MVP from PRD/architecture."""
from __future__ import annotations

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


def resolve_app_mode(requirement: str, prd: str = "", uiux: str = "") -> str:
    if is_doc_chat_domain(requirement, prd, uiux):
        return "doc_chat"
    if is_ecommerce_domain(requirement, prd, uiux):
        return "ecommerce"
    if is_quality_domain(requirement, prd, uiux):
        return "quality"
    return "generic"


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
    elif mode == "quality":
        files.update(frontend_files(title, colors, True))
        files.update(backend_files(title, True))
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

    if mode in ("doc_chat", "ecommerce"):
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
