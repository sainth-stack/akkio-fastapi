from app_builder.services.sap_community_catalog import catalog_best_match
from app_builder.services.sap_community_rag import _search_query_from_ticket, build_ticket_rag_record
from app_builder.services.sap_community_search import search_sap_community_first


def test_search_query_from_ticket_uses_title():
    q = _search_query_from_ticket(
        {"title": "IDoc failure in MM goods receipt", "description": "status 51", "component": "Other"}
    )
    assert "IDoc failure" in q
    assert "status 51" not in q or "IDoc" in q


def test_each_ticket_gets_own_rag():
    row_a = {"ticket_id": "A", "title": "Account determination ID missing in master data contract account (FI-CA)"}
    row_b = {"ticket_id": "B", "title": "BAPI timeout on sales order create"}
    rag_a = build_ticket_rag_record(row_a, force=True)
    rag_b = build_ticket_rag_record(row_b, force=True)
    assert rag_a["search_query"] != rag_b["search_query"]
    assert rag_a["suggested_answer"] != rag_b["suggested_answer"]
    assert "0UCACTDETID" in rag_a["suggested_answer"] or "Account Determination" in rag_a["suggested_answer"]
    assert "SM59" in rag_b["suggested_answer"] or "RFC" in rag_b["suggested_answer"] or "BAPI" in rag_b["suggested_answer"]


def test_catalog_match_account_determination():
    hit = search_sap_community_first("account determination id in DOCUBATCH", live=False)
    assert hit.source == "catalog"
    assert "DOCUBATCH" in hit.title or "docubatch" in hit.title.lower()


def test_catalog_best_match_scores():
    entry = catalog_best_match("FI-AR Account determination customer reconciliation")
    assert entry
    assert "fi-ar" in str(entry.get("title") or "").lower() or "fi-ar" in str(entry.get("keywords"))