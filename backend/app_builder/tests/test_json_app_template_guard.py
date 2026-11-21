import json

from app_builder.services.json_app_template_guard import (
    ensure_json_app_renderer,
    is_json_app_project,
    repair_broken_legacy_app_tsx,
)


def _json_app_files(app_tsx: str) -> dict:
    return {
        "frontend/src/mock/data.json": json.dumps(
            {
                "version": 1,
                "pages": [{"id": "dashboard"}],
                "navigation": [{"pageId": "dashboard", "path": "/", "label": "Dashboard"}],
            }
        ),
        "frontend/src/App.tsx": app_tsx,
        "frontend/package.json": '{"dependencies":{"@mui/material":"6"}}',
    }


def test_detects_json_app():
    assert is_json_app_project(_json_app_files("export default function App() { return null; }"))


def test_replaces_legacy_reports_app():
    broken = """
export default function App() {
  return <ReportsPage />;
}
"""
    files = _json_app_files(broken)
    fixed, ok = repair_broken_legacy_app_tsx(files)
    assert ok
    assert "PageRenderer" in fixed["frontend/src/App.tsx"]
    assert "ReportsPage" not in fixed["frontend/src/App.tsx"]


def test_ensure_restores_renderer():
    files = _json_app_files("<ReportsPage />")
    out, n = ensure_json_app_renderer(files)
    assert n > 0
    assert "PageRenderer" in out["frontend/src/App.tsx"]
