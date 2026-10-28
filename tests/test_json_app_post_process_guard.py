from app_builder.services.code_post_process import post_process_generated_files
from app_builder.services.fullstack_codegen import is_fullstack_generated_files


def _json_app_files(app_tsx: str) -> dict:
    return {
        "frontend/src/mock/data.json": (
            '{"version":1,"pages":[{"id":"dashboard"},{"id":"reports"}],"navigation":'
            '[{"pageId":"dashboard","path":"/"},{"pageId":"reports","path":"/reports"}],'
            '"meta":{"appName":"Test"}}'
        ),
        "frontend/src/App.tsx": app_tsx,
        "frontend/package.json": '{"dependencies":{"@mui/material":"^6.0.0","react":"18"}}',
    }


def test_json_app_not_treated_as_fullstack():
    files = _json_app_files("<ReportsPage />")
    assert not is_fullstack_generated_files(files)


def test_post_process_restores_page_renderer():
    broken = _json_app_files(
        "export default function App(){ return <ReportsPage />; }"
    )
    out = post_process_generated_files(
        broken,
        requirement="manufacturing dashboard",
        prd="Dashboard Analytics Reports",
        builder_kind="fullstack",
    )
    assert "PageRenderer" in out["frontend/src/App.tsx"]
    assert "ReportsPage" not in out["frontend/src/App.tsx"]
