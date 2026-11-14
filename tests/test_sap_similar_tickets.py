from app_builder.schemas.app_definition import AppDefinition, AppMeta, NavItemDef, PageDef
from app_builder.services.sap_similar_tickets import (
    ensure_sap_similar_ticket_features,
    find_similar_tickets,
    parse_sap_ticket_count,
    seed_sap_tickets,
    seed_sap_tickets_if_empty,
    wants_sap_similar_tickets,
)


def test_wants_sap_prompt():
    assert wants_sap_similar_tickets("", "Build SAP similar tickets app with create and match")
    assert wants_sap_similar_tickets("", "Find similar ticket for SAP IDoc errors")


def test_parse_ticket_count_from_prompt():
    assert parse_sap_ticket_count("", "Seed 10 SAP community tickets with RAG") == 10
    assert parse_sap_ticket_count("SAP app", "Dashboard with 25 tickets") == 25
    assert parse_sap_ticket_count("", "no count here") is None


def test_seed_respects_prompt_count():
    defn = seed_sap_tickets({"data": {}}, requirement="generate 12 synthetic sap tickets")
    assert defn["meta"]["sapTicketSeedCount"] == 12
    assert len(defn["data"]["source_sap_tickets"]) == 12


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


def test_ensure_when_blueprint_domain_sap_without_keywords():
    defn = AppDefinition(
        meta=AppMeta(appName="SAP Desk", description="", domain="sap"),
        theme={},
        navigation=[],
        pages=[],
    )
    out = ensure_sap_similar_ticket_features(defn, prd="", requirement="Dashboard with KPIs only")
    assert out.meta.domain == "sap"
    assert "sap_similar_tickets" in (out.meta.capabilities or [])


def test_sap_single_dashboard_nav_only():
    defn = AppDefinition(
        meta=AppMeta(appName="SAP Desk", description="", domain="generic"),
        theme={},
        navigation=[
            NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard"),
            NavItemDef(label="Ticket List", path="/tickets", icon="List", pageId="ticket_list"),
            NavItemDef(label="Predict", path="/predict", icon="TrendingUp", pageId="predict"),
        ],
        pages=[
            PageDef(id="dashboard", title="Dashboard", blocks=[]),
            PageDef(id="ticket_list", title="Ticket List", blocks=[]),
            PageDef(id="predict", title="Predict", blocks=[]),
        ],
    )
    out = ensure_sap_similar_ticket_features(defn, prd="SAP", requirement="similar tickets dashboard")
    assert len(out.navigation) == 1
    assert out.navigation[0].pageId == "dashboard"
    assert len(out.pages) == 1
    assert out.pages[0].id == "dashboard"
    block_types = [b.get("type") for b in out.pages[0].blocks if isinstance(b, dict)]
    assert block_types == ["kpiRow", "dataTable"]
    assert "predict" not in (out.meta.capabilities or [])


def test_default_sap_dashboard_not_two_screens():
    defn = AppDefinition(
        meta=AppMeta(appName="Similarity Tickets", description="", domain="generic"),
        theme={},
        navigation=[],
        pages=[],
    )
    out = ensure_sap_similar_ticket_features(
        defn,
        prd="SAP similarity tickets",
        requirement="Dashboard with KPIs. Seed 15 SAP community tickets. Chatbot finds similar tickets.",
    )
    assert out.meta.sapTicketSeedCount == 15
    assert out.meta.domain == "sap"
    assert "rag_chat" in (out.meta.capabilities or [])
    assert (out.meta.sapTicketSeedCount or 0) >= 1
    assert any(p.id == "dashboard" for p in out.pages)
    dash = next(p for p in out.pages if p.id == "dashboard")
    types = [b.get("type") for b in dash.blocks if isinstance(b, dict)]
    assert "kpiRow" in types
    assert "dataTable" in types
    assert "sapCommunityFeed" not in types


def test_ticket_rag_enrich():
    from app_builder.services.sap_community_rag import enrich_definition_tickets

    defn = seed_sap_tickets_if_empty({"data": {}})
    out = enrich_definition_tickets(defn)
    row = out["data"]["source_sap_tickets"][0]
    assert row.get("snippet")
    assert row.get("suggested_answer")
    assert out["data"]["sap_ticket_rag"]


def test_merge_replaces_community_feed_with_table():
    defn = AppDefinition(
        meta=AppMeta(appName="Sap Similarity", description="", domain="sap"),
        theme={},
        navigation=[NavItemDef(label="Dashboard", path="/", icon="Dashboard", pageId="dashboard")],
        pages=[
            PageDef(
                id="dashboard",
                title="Dashboard",
                blocks=[
                    {"type": "kpiRow", "dataKey": "kpis_dashboard", "items": []},
                    {"type": "sapCommunityFeed", "dataKey": "source_sap_tickets"},
                ],
            )
        ],
    )
    out = ensure_sap_similar_ticket_features(defn, prd="SAP", requirement="dashboard KPIs table only")
    dash = next(p for p in out.pages if p.id == "dashboard")
    types = [b.get("type") for b in dash.blocks if isinstance(b, dict)]
    assert "sapCommunityFeed" not in types
    assert "dataTable" in types


def test_wants_sap_rag_step2_prompt():
    from app_builder.services.sap_rag_update import wants_sap_rag_followup

    step2 = (
        "Step 2 — load SAP ticket data and context.\n"
        "For each ticket, search SAP Community and build per-ticket RAG."
    )
    assert wants_sap_rag_followup(step2)


def test_sap_chat_account_determination():
    from app_builder.services.sap_ticket_chat import answer_sap_ticket_chat

    defn = seed_sap_tickets_if_empty({"meta": {"domain": "sap", "capabilities": ["sap_similar_tickets"]}})
    out = answer_sap_ticket_chat(defn, [{"role": "user", "content": "account determination id"}])
    assert out
    body = out["content"]
    assert "Problem Description" in body
    assert "RETMFG" in body
    assert "Suggested Solution" in body
