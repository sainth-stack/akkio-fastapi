from app_builder.schemas.app_definition import AppDefinition, AppMeta, NavItemDef, PageDef
from app_builder.services.intelligence_pages import ensure_intelligence_features


def _minimal_dashboard() -> AppDefinition:
    return AppDefinition(
        meta=AppMeta(appName="Ops Monitor", description="", domain="manufacturing"),
        theme={},
        navigation=[NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard")],
        pages=[PageDef(id="dashboard", title="Dashboard", blocks=[])],
    )


def test_adds_predict_nav_and_capabilities():
    defn = _minimal_dashboard()
    out = ensure_intelligence_features(defn, prd="maintenance dashboard with RAG chat", requirement="")
    assert "predict" in out.meta.capabilities
    assert any(n.pageId == "predict" for n in out.navigation)
    assert any(p.id == "predict" for p in out.pages)


def test_skips_ecommerce():
    defn = _minimal_dashboard()
    defn.meta = defn.meta.model_copy(update={"domain": "ecommerce", "appName": "Shop"})
    out = ensure_intelligence_features(defn, prd="predict and rag", requirement="")
    assert not any(n.pageId == "predict" for n in out.navigation)
