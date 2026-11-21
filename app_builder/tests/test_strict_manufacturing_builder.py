import asyncio
import copy
import json
import sys
from types import SimpleNamespace

import pytest

from app_builder.schemas.app_definition import AppDefinition
from app_builder.schemas.app_definition_llm import PageBlocksPlan
from app_builder.services.definition_contract import validate_definition, validate_page_data
from app_builder.services.definition_runtime import load_definition, mutate_record, save_definition
from app_builder.services.prd_structure_parser import section_for_page


def definition():
    return {
        "meta": {"appName": "Casting Ops", "domain": "casting", "generationPolicy": "strict_llm_v1", "generatedBy": "llm_pipeline", "dataModel": "definition_v1"},
        "navigation": [{"label": "Reports", "path": "/", "pageId": "reports"}],
        "pages": [{"id": "reports", "title": "Reports", "blocks": [
            {"type": "kpiRow", "dataKey": "kpis_reports", "items": [{"label": "Rejected castings", "valueKey": "rejected"}]},
            {"type": "dataTable", "dataKey": "castings", "columns": [{"field": "batch", "header": "Batch", "required": True}, {"field": "status", "header": "Status"}]},
            {"type": "actionBar", "actions": [{"id": "create", "label": "Create batch", "actionType": "create", "dataKey": "castings"}, {"id": "export", "label": "Export CSV", "actionType": "exportCsv"}]},
        ]}],
        "mock": {"kpis_reports": {"rejected": 0}, "castings": [{"batch": "CAST-72", "status": "Closed"}]},
    }


def test_nested_prd_headings_preserved():
    prd = "### Reports\nPurpose: batch review\n#### KPIs\n- Rejected castings\n#### Table columns\n- Batch\n### Scheduling\n- unrelated"
    section = section_for_page(prd, "Reports", "reports")
    assert "Rejected castings" in section
    assert "Batch" in section
    assert "unrelated" not in section


def test_strict_finalization_preserves_layout_status_and_zero():
    from app_builder.services.app_definition_generator import finalize_app_definition

    model = AppDefinition.model_validate(definition())
    before = copy.deepcopy(model.pages)
    data = copy.deepcopy(model.mock)
    result = finalize_app_definition(model, requirement="Casting reports with KPI, table, create and export")
    assert result.pages == before
    assert result.mock == data
    assert result.mock["castings"][0]["status"] == "Closed"
    assert result.meta.promptHash


def test_runtime_compute_does_not_seed_or_overwrite_strict_data():
    from app_builder.services.operational_metrics_compute import apply_source_data_pipeline, recompute_derived_data

    payload = definition()
    before = copy.deepcopy(payload)
    assert apply_source_data_pipeline(payload, seed_if_empty=True, recompute_from_source=True) == before
    assert recompute_derived_data(payload) == before


@pytest.mark.parametrize("data", [{}, {"castings": [{"batch": "B"}], "kpis_reports": {"rejected": 0}}])
def test_missing_data_is_rejected(data):
    with pytest.raises(ValueError):
        validate_page_data(definition()["pages"][0]["blocks"], data)


def test_empty_dataset_is_not_fabricated():
    payload = definition()
    payload["mock"]["castings"] = []
    validate_definition(AppDefinition.model_validate(payload))
    assert payload["mock"]["castings"] == []


def test_unknown_block_rejected_and_single_widget_allowed():
    PageBlocksPlan(blocks=[{"type": "text", "content": "Instructions requested by user"}])
    with pytest.raises(ValueError):
        PageBlocksPlan(blocks=[{"type": "unknown"}])


def test_record_persistence_and_validation(tmp_path, monkeypatch):
    import app_builder.services.definition_runtime as runtime

    monkeypatch.setattr(runtime, "resolve_project_root", lambda project: str(tmp_path))
    original = definition()
    updated = mutate_record(original, "castings", {"batch": "CAST-73", "status": "Scheduled"})
    save_definition("casting", updated)
    assert load_definition("casting") == updated
    assert len(original["mock"]["castings"]) == 1
    assert len(updated["data"]["castings"]) == 2
    with pytest.raises(ValueError):
        mutate_record(updated, "unknown", {})
    with pytest.raises(ValueError):
        mutate_record(updated, "castings", {"batch": "", "status": "Scheduled"})


def test_llm_pipeline_uses_selected_model_and_preserves_response(monkeypatch):
    from app_builder.services.app_definition_llm_pipeline import generate_app_definition_llm

    calls = []
    payload = definition()
    responses = [
        {"appName": "Casting Ops", "domain": "casting", "navigation": payload["navigation"], "pages": [{"id": "reports", "title": "Reports"}]},
        {"blocks": payload["pages"][0]["blocks"]},
        {"data": payload["mock"]},
    ]

    class LLM:
        async def ainvoke(self, messages):
            calls.append(messages[-1].content)
            return SimpleNamespace(content=json.dumps(responses.pop(0)))

    selected = []
    def get_llm(user, **kwargs):
        selected.append((user, kwargs.get("model_name")))
        return LLM()

    monkeypatch.setitem(sys.modules, "llm_helper", SimpleNamespace(get_llm_for_user=get_llm))
    result = asyncio.run(generate_app_definition_llm(requirement="Casting reports", user_email="test@example.com", model_name="selected-model"))
    assert selected == [("test@example.com", "selected-model")]
    assert result.mock == payload["mock"]
    assert result.pages[0].blocks == payload["pages"][0]["blocks"]
    assert "Casting reports" in calls[-1]


def test_failed_page_generation_has_no_static_fallback(monkeypatch):
    from app_builder.services.app_definition_llm_pipeline import generate_app_definition_llm

    responses = [{"appName": "Casting", "navigation": [{"label": "Reports", "path": "/", "pageId": "reports"}], "pages": [{"id": "reports", "title": "Reports"}]}, {}, {}, {}]

    class LLM:
        async def ainvoke(self, messages):
            return SimpleNamespace(content=json.dumps(responses.pop(0)))

    monkeypatch.setitem(sys.modules, "llm_helper", SimpleNamespace(get_llm_for_user=lambda *args, **kwargs: LLM()))
    with pytest.raises(ValueError, match="LLM failed to generate blocks"):
        asyncio.run(generate_app_definition_llm(requirement="Casting reports"))


def test_source_bound_metrics_recompute_without_guessed_values():
    from app_builder.services.definition_metrics import compute_definition

    payload = definition()
    block = payload["pages"][0]["blocks"][0]
    block["sourceDataKey"] = "castings"
    block["items"] = [{"label": "Open batches", "valueKey": "open", "aggregation": "count", "where": {"status": "Open"}}]
    result = compute_definition(payload)
    assert result["mock"]["kpis_reports"]["open"] == 0
    result = mutate_record(result, "castings", {"batch": "CAST-73", "status": "Open"})
    assert result["data"]["kpis_reports"]["open"] == 1


def test_unknown_metric_field_fails_instead_of_fabricating():
    from app_builder.services.definition_metrics import compute_definition

    payload = definition()
    block = payload["pages"][0]["blocks"][0]
    block["sourceDataKey"] = "castings"
    block["items"][0].update({"aggregation": "avg", "field": "oee"})
    with pytest.raises(ValueError, match="unknown fields"):
        compute_definition(payload)


def test_strict_csv_import_preserves_rows_and_recomputes(monkeypatch):
    from app_builder.services.app_definition_generator import finalize_app_definition
    from app_builder.services.data_import_service import merge_upload_bytes

    payload = definition()
    payload["pages"][0]["blocks"][0].update({"sourceDataKey": "castings", "items": [{"label": "Open batches", "valueKey": "open", "aggregation": "count", "where": {"status": "Open"}}]})
    payload["mock"]["kpis_reports"] = {"open": 0}
    model = finalize_app_definition(AppDefinition.model_validate(payload))
    updated, summary = merge_upload_bytes(model.model_dump(), filename="castings.csv", raw=b"Batch,Status\nCAST-99,Open\n", target_key="castings")
    assert updated["data"]["castings"] == [{"batch": "CAST-99", "status": "Open"}]
    assert updated["data"]["kpis_reports"]["open"] == 1
    assert summary["rowCount"] == 1