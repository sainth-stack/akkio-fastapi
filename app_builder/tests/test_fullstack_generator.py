from app_builder.services.fullstack_app_generator import (
    generate_fullstack_application,
    fill_missing_fullstack_files,
    is_quality_domain,
    resolve_app_mode,
)
from app_builder.services.fullstack_docchat_generator import is_doc_chat_domain
from app_builder.services.fullstack_codegen import is_fullstack_generated_files
from app_builder.services.code_post_process import post_process_generated_files
from app_builder.services.project_config import _extract_tables_from_contract, build_project_config


REQUIREMENT = '''
Build "Supplier Quality & Incoming Material Release"
incoming lot CAPA ppm release decision supplier quality
'''


def test_quality_domain_detection():
    assert is_quality_domain(REQUIREMENT)


def test_doc_chat_two_screens():
    req = "create two screens\n1. upload a files\n2. chat with that if user asks questions answer from doc"
    assert is_doc_chat_domain(req)
    assert resolve_app_mode(req) == "doc_chat"
    files = generate_fullstack_application(req, req, "", {})
    assert "UploadPage" in files["frontend/src/App.tsx"]
    assert "Document chat" in files["frontend/src/layout/AppShell.tsx"]
    assert "/api/documents/upload" in files["backend/routes.py"]
    assert "/api/chat/ask" in files["backend/routes.py"]


def test_generator_emits_saas_modules():
    files = generate_fullstack_application(REQUIREMENT, REQUIREMENT, REQUIREMENT, {})
    assert "frontend/src/App.jsx" not in files
    assert "frontend/src/App.tsx" in files
    assert files["frontend/src/App.tsx"].count("<Route") >= 10
    assert "Incoming Lots" in files["frontend/src/layout/AppShell.tsx"]
    assert "TrendChart" in files["frontend/src/pages/DashboardPage.tsx"]
    assert "class Supplier" in files["backend/models.py"]
    assert "/api/suppliers" in files["backend/routes.py"]
    assert "/api/ai/ask" in files["backend/routes.py"]
    assert "LOT-20260925" in files["backend/seed.py"]
    assert "ABC Auto Components" in files["frontend/src/api/mock.ts"]
    assert "KPICard" in files["frontend/src/components/KPICard.tsx"]
    assert "docker-compose.yml" in files
    assert "pytest" in files["backend/requirements.txt"]
    assert "alembic/versions/001_initial_schema.py" in "".join(files.keys()) or "backend/alembic/versions/001_initial_schema.py" in files


def test_fill_missing_replaces_thin_llm_shell():
    thin = {
        "frontend/src/App.jsx": "export default function App(){return <div/>}",
        "frontend/src/styles/app.css": "body{}",
        "frontend/src/App.tsx": "export default function App(){ return <div/> }",
        "backend/routes.py": "from fastapi import APIRouter\nrouter=APIRouter()",
    }
    out = fill_missing_fullstack_files(thin, REQUIREMENT, REQUIREMENT, REQUIREMENT, {})
    assert "frontend/src/App.jsx" not in out
    assert out["frontend/src/App.tsx"].count("<Route") >= 10
    assert "/api/suppliers" in out["backend/routes.py"]
    assert "Incoming Lots" in out["frontend/src/layout/AppShell.tsx"]


def test_post_process_keeps_fullstack_file_count():
    files = generate_fullstack_application(REQUIREMENT, REQUIREMENT, REQUIREMENT, {})
    before = len(files)
    out = post_process_generated_files(files, builder_kind="fullstack")
    assert len(out) >= before - 2
    assert "frontend/src/App.tsx" in out
    assert out["frontend/src/App.tsx"].count("<Route") >= 10
    assert "frontend/src/pages/DashboardPage.tsx" in out
    assert "Incoming Lots" in out.get("frontend/src/layout/AppShell.tsx", "")
    assert is_fullstack_generated_files(out)
    assert "shell-notice" not in (out.get("frontend/src/App.jsx") or "").lower()


def test_list_of_dict_api_contract_does_not_split_dict():
    contract = {
        "endpoints": [
            {"method": "GET", "path": "/api/suppliers"},
            {"method": "POST", "path": "/api/inspections"},
            {"method": "GET", "path": "/api/incoming-lots/{id}"},
        ]
    }
    tables = _extract_tables_from_contract(contract)
    assert "suppliers" in tables
    assert "inspections" in tables
    config = build_project_config(
        project_name="sqms-test",
        structured_requirement={"project_name": "sqms-test", "description": REQUIREMENT, "entities": []},
        architecture={"api_contract": contract},
        api_contract=contract,
        db_schema={"schema": ""},
        template_name="base-fullstack-vite-mui",
    )
    assert config
