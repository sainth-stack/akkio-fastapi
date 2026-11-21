from app_builder.services.fullstack_stack import lock_architecture, is_fullstack
from app_builder.services.fullstack_codegen import run_screen_qa, get_fullstack_scaffold_files


def test_is_fullstack():
    assert is_fullstack("fullstack")
    assert not is_fullstack("app")
    assert not is_fullstack(None)


def test_lock_architecture_forces_stack():
    arch = lock_architecture({"frontend_structure": {"ui_library": "Tailwind"}, "backend_structure": {"database": "SQLite"}})
    assert arch["frontend_structure"]["ui_library"] == "Material UI (MUI)"
    assert arch["frontend_structure"]["language"] == "TypeScript"
    assert arch["backend_structure"]["database"] == "PostgreSQL"
    assert arch["backend_structure"]["framework"] == "FastAPI"


def test_fullstack_scaffold_exists():
    files = get_fullstack_scaffold_files()
    assert "frontend/package.json" in files
    assert "frontend/src/api/client.ts" in files
    assert "mockFetch" in files["frontend/src/api/client.ts"]
    assert "docker-compose.yml" in files
    assert "backend/main.py" in files
    assert "@mui/material" in files["frontend/package.json"]
    assert "postgresql" in files["docker-compose.yml"]


def test_screen_qa_detects_dashboard():
    files = {"frontend/src/pages/DashboardPage.tsx": "export default function Dashboard() { return <div>Dashboard</div> }"}
    passed, missing = run_screen_qa(files, {"frontend_structure": {"key_components": ["Dashboard"]}})
    assert "Dashboard" in passed
    assert "Dashboard" not in missing
