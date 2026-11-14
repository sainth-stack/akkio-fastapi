"""Search SAP Community for the first hit per ticket query (Bing/CSE or catalog)."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import quote_plus

import requests

from app_builder.services.sap_community_catalog import catalog_best_match

logger = logging.getLogger("app_builder")

_SEARCH_TIMEOUT = float(os.getenv("SAP_COMMUNITY_SEARCH_TIMEOUT", "12"))
_USER_AGENT = "Akkio-SAP-Ticket-RAG/1.0 (+https://akkio.com; research bot)"


@dataclass
class CommunitySearchHit:
    url: str
    title: str
    snippet: str
    body: str
    suggested_answer: str
    product: str
    source: str  # bing | google_cse | catalog | fallback


def _bing_search(query: str) -> Optional[CommunitySearchHit]:
    key = (os.getenv("BING_SEARCH_API_KEY") or os.getenv("AZURE_BING_SEARCH_KEY") or "").strip()
    if not key:
        return None
    q = f"site:community.sap.com {query}"
    try:
        res = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers={"Ocp-Apim-Subscription-Key": key},
            params={"q": q, "count": 5, "mkt": "en-US"},
            timeout=_SEARCH_TIMEOUT,
        )
        res.raise_for_status()
        data = res.json()
        pages = (data.get("webPages") or {}).get("value") or []
        for page in pages:
            url = str(page.get("url") or "")
            if "community.sap.com" not in url:
                continue
            title = str(page.get("name") or query)
            snippet = str(page.get("snippet") or "")
            body = _fetch_community_text(url) or snippet
            suggested = _suggested_from_body(body, title)
            return CommunitySearchHit(
                url=url,
                title=title,
                snippet=snippet[:400] or title[:200],
                body=body,
                suggested_answer=suggested,
                product="SAP ERP",
                source="bing",
            )
    except Exception as exc:
        logger.warning("[sap_search] bing failed: %s", exc)
    return None


def _google_cse_search(query: str) -> Optional[CommunitySearchHit]:
    api_key = (os.getenv("GOOGLE_CSE_API_KEY") or "").strip()
    cx = (os.getenv("GOOGLE_CSE_CX") or "").strip()
    if not api_key or not cx:
        return None
    q = f"site:community.sap.com {query}"
    try:
        res = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cx, "q": q, "num": 5},
            timeout=_SEARCH_TIMEOUT,
        )
        res.raise_for_status()
        items = res.json().get("items") or []
        for item in items:
            url = str(item.get("link") or "")
            if "community.sap.com" not in url:
                continue
            title = str(item.get("title") or query)
            snippet = str(item.get("snippet") or "")
            body = _fetch_community_text(url) or snippet
            suggested = _suggested_from_body(body, title)
            return CommunitySearchHit(
                url=url,
                title=title,
                snippet=snippet[:400],
                body=body,
                suggested_answer=suggested,
                product="SAP ERP",
                source="google_cse",
            )
    except Exception as exc:
        logger.warning("[sap_search] google cse failed: %s", exc)
    return None


def _fetch_community_text(url: str) -> str:
    if not url or "community.sap.com" not in url:
        return ""
    try:
        res = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html"},
            timeout=_SEARCH_TIMEOUT,
        )
        if res.status_code >= 400:
            return ""
        html = res.text
        if "Just a moment" in html or "cloudflare" in html.lower():
            return ""
        text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:6000]
    except Exception as exc:
        logger.debug("[sap_search] fetch skip %s: %s", url, exc)
        return ""


def _suggested_from_body(body: str, title: str) -> str:
    body = (body or "").strip()
    if not body:
        return f"See SAP Community thread: {title}"
    sentences = re.split(r"(?<=[.!?])\s+", body)
    lines = [s for s in sentences if len(s) > 20][:6]
    if not lines:
        lines = [body[:800]]
    return "\n".join(f"• {s.strip()}" for s in lines[:5])


def _hit_from_catalog(query: str) -> CommunitySearchHit:
    entry = catalog_best_match(query) or {}
    body = str(entry.get("body") or entry.get("snippet") or "")
    return CommunitySearchHit(
        url=str(entry.get("url") or ""),
        title=str(entry.get("title") or query),
        snippet=str(entry.get("snippet") or body[:220]),
        body=body,
        suggested_answer=str(entry.get("suggested_answer") or _suggested_from_body(body, query)),
        product=str(entry.get("product") or "SAP ERP"),
        source="catalog",
    )


def search_sap_community_first(
    query: str,
    *,
    live: bool = True,
) -> CommunitySearchHit:
    """Return first Community hit for this ticket's query (unique per ticket title)."""
    q = (query or "").strip()
    if not q:
        q = "SAP ERP incident"
    if live:
        for fn in (_bing_search, _google_cse_search):
            hit = fn(q)
            if hit:
                return hit
    return _hit_from_catalog(q)


def community_search_link(query: str) -> str:
    return (
        "https://community.sap.com/t5/forums/searchpage/tab/message?"
        f"advanced=false&allow_punctuation=true&q={quote_plus(query[:120])}"
    )