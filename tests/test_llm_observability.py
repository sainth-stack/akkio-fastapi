from app_builder.schemas.app_definition import AppDefinition, AppMeta, NavItemDef, PageDef
from app_builder.services.llm_model_pricing import estimate_cost_usd, extract_token_breakdown
from app_builder.services.llm_observability_pages import ensure_observability_features


def _minimal_dashboard() -> AppDefinition:
    return AppDefinition(
        meta=AppMeta(appName="Ops", description="", domain="manufacturing"),
        theme={},
        navigation=[NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard")],
        pages=[PageDef(id="dashboard", title="Dashboard", blocks=[])],
    )


class _FakeUsage:
    prompt_tokens = 100
    completion_tokens = 50


class _FakeResp:
    usage = _FakeUsage()


def test_wants_observability_adds_nav_and_capability():
    defn = _minimal_dashboard()
    out = ensure_observability_features(
        defn,
        prd="Dashboard with chat and LLM observability for token usage",
        requirement="",
    )
    assert "llm_observability" in out.meta.capabilities
    assert any(n.pageId == "observability" for n in out.navigation)
    assert any(p.id == "observability" for p in out.pages)


def test_extract_tokens_and_cost():
    usage = extract_token_breakdown(_FakeResp())
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 150
    cost = estimate_cost_usd("gpt-4o-mini", 100, 50)
    assert cost > 0
