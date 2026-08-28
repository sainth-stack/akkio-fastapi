"""Tests for functionality validator (apiFetch fixes, CRUD smoke test)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app_builder.services.functionality_validator import (
    fix_api_fetch_signature,
    validate_and_fix_functionality,
    enrich_project_config,
    verify_runtime_crud,
)
from app_builder.services.app_spec_service import build_app_spec


BROKEN_APP = """
import { apiFetch } from './api/client.js';
export default function App() {
  const create = async () => {
    await apiFetch('/tasks', 'POST', { title: 'x' });
    await apiFetch(`/tasks/${1}`, 'PUT', { completed: true });
    await apiFetch(`/tasks/${1}`, 'DELETE');
  };
}
"""


def test_fix_api_fetch_signature():
    fixed, n = fix_api_fetch_signature(BROKEN_APP)
    assert n == 3
    assert "method: 'POST'" in fixed
    assert "JSON.stringify({ title: 'x' })" in fixed
    assert "'POST'" not in fixed.split("method:")[0]  # no raw method string arg


def test_validate_and_fix_no_errors_after_fix():
    arch = {"database_schema": {"tables": [{"name": "tasks", "columns": [{"name": "title"}, {"name": "completed"}]}]}}
    spec = build_app_spec("todo app", arch, "tasks")
    files = {"frontend/src/App.jsx": BROKEN_APP}
    out, errs, msgs = validate_and_fix_functionality(files, arch, spec)
    assert not errs, errs
    assert any("apiFetch" in m for m in msgs)


def test_enrich_project_config_fields():
    config = {"projectId": "test_proj", "tables": ["tasks"], "fields": {}}
    arch = {"database_schema": {"tables": [{"name": "tasks", "columns": [{"name": "title"}, {"name": "completed"}]}]}}
    out = enrich_project_config(config, arch)
    assert out["fields"]["tasks"] == ["title", "completed"]


if __name__ == "__main__":
    test_fix_api_fetch_signature()
    test_validate_and_fix_no_errors_after_fix()
    test_enrich_project_config_fields()
    print("ALL OK")
