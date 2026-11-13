from app_builder.schemas.app_definition import AppDefinition, AppMeta, NavItemDef, PageDef
from app_builder.services.sap_similar_tickets import (
    ensure_sap_similar_ticket_features,
    find_similar_tickets,
    seed_sap_tickets_if_empty,
    wants_sap_similar_tickets,
)


def test_wants_sap_prompt():
    assert wants_sap_similar_tickets("", "Build SAP similar tickets app with create and match")
    assert wants_sap_similar_tickets("", "Find similar ticket for SAP IDoc errors")


def test_seed_and_similarity():
    defn = seed_sap_tickets_if_empty({"data": {}})
    rows = defn["data"]["source_sap_tickets"]
    assert len(rows) >= 6
    ref = {
        "title": "IDoc failure in MM goods receipt",
        "description": "Inbound IDoc 740 failed with status 51",
        "category": "Materials Management",
        "component": "IDoc",
    }
    similar = find_similar_tickets(ref, rows, limit=3)
    assert similar
    assert any("idoc" in str(s.get("component", "")).lower() for s in similar)


def test_ensure_two_screens():
    defn = AppDefinition(
        meta=AppMeta(appName="SAP Desk", description="", domain="generic"),
        theme={},
        navigation=[NavItemDef(label="Home", path="/", icon="Dashboard", pageId="dashboard")],
        pages=[PageDef(id="dashboard", title="Dashboard", blocks=[])],
    )
    out = ensure_sap_similar_ticket_features(
        defn,
        prd="SAP domain",
        requirement="Two screens only: Create Ticket and Similar Tickets for similar ticket search",
    )
    assert out.meta.domain == "sap"
    assert "sap_similar_tickets" in out.meta.capabilities
    ids = [p.id for p in out.pages]
    assert ids == ["create_ticket", "similar_tickets"]
