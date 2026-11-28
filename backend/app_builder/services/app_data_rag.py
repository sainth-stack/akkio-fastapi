"""RAG-style context from generated app definition + datasets for in-app chat."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from app_builder.services.runtime_paths import resolve_project_root

_MAX_CONTEXT_CHARS = 28_000
_MAX_TABLE_SAMPLE_ROWS = 5
_MAX_CHART_POINTS = 8


def load_project_app_definition(project_id: str) -> Optional[Dict[str, Any]]:
    root = resolve_project_root(project_id)
    if not os.path.isdir(root):
        return None
    for rel in (
        "frontend/src/data/store-context.json",
        "frontend/src/mock/data.json",
        "frontend/src/config/app-definition.json",
        "app_definition.json",
    ):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _data_bag(defn: Dict[str, Any]) -> Dict[str, Any]:
    bag = defn.get("data") or defn.get("mock") or {}
    return bag if isinstance(bag, dict) else {}


def _numeric_stats(values: List[float]) -> str:
    if not values:
        return "no numeric points"
    return (
        f"min={min(values):.2f}, max={max(values):.2f}, "
        f"avg={sum(values) / len(values):.2f}, latest={values[-1]:.2f}"
    )


def _summarize_chart_block(block: Dict[str, Any], data: Dict[str, Any]) -> str:
    title = str(block.get("title") or "Chart").strip()
    dk = str(block.get("dataKey") or "")
    rows = data.get(dk)
    if not isinstance(rows, list) or not rows:
        return f"- {title}: (no data for key {dk})"
    series = block.get("series") or []
    series_keys = [
        str(s.get("dataKey"))
        for s in series
        if isinstance(s, dict) and s.get("dataKey")
    ]
    x_key = str(block.get("xAxisKey") or "period")
    lines = [f"- {title} ({block.get('type', 'chart')}, dataKey={dk}):"]
    if not series_keys and isinstance(rows[0], dict):
        series_keys = [k for k in rows[0].keys() if k != x_key][:6]
    for sk in series_keys[:6]:
        nums: List[float] = []
        for row in rows[:_MAX_CHART_POINTS]:
            if not isinstance(row, dict):
                continue
            val = row.get(sk)
            if isinstance(val, (int, float)):
                nums.append(float(val))
        if nums:
            lines.append(f"  • {sk}: {_numeric_stats(nums)}")
        else:
            sample = [
                str(row.get(sk, ""))[:40]
                for row in rows[:3]
                if isinstance(row, dict)
            ]
            lines.append(f"  • {sk}: samples {sample}")
    if isinstance(rows[0], dict) and x_key in rows[0]:
        labels = [str(r.get(x_key, ""))[:24] for r in rows[:_MAX_CHART_POINTS] if isinstance(r, dict)]
        lines.append(f"  • x-axis ({x_key}): {labels}")
    return "\n".join(lines)


def _summarize_table_block(block: Dict[str, Any], data: Dict[str, Any]) -> str:
    title = str(block.get("title") or "Table").strip()
    dk = str(block.get("dataKey") or "")
    rows = data.get(dk)
    if not isinstance(rows, list):
        return f"- {title}: (no rows for {dk})"
    cols = block.get("columns") or []
    fields = [
        str(c.get("field"))
        for c in cols
        if isinstance(c, dict) and c.get("field")
    ]
    if not fields and rows and isinstance(rows[0], dict):
        fields = list(rows[0].keys())[:12]
    lines = [f"- {title} ({len(rows)} rows, dataKey={dk})"]
    headers = {
        str(c.get("field")): str(c.get("header") or c.get("field"))
        for c in cols
        if isinstance(c, dict) and c.get("field")
    }
    if fields:
        lines.append("  Columns: " + ", ".join(headers.get(f, f) for f in fields[:12]))
    for i, row in enumerate(rows[:_MAX_TABLE_SAMPLE_ROWS]):
        if not isinstance(row, dict):
            continue
        cells = [f"{headers.get(f, f)}={row.get(f)}" for f in fields[:8] if f in row]
        lines.append(f"  Row {i + 1}: " + "; ".join(cells))
    return "\n".join(lines)


def _summarize_kpi_block(block: Dict[str, Any], data: Dict[str, Any], page_id: str) -> str:
    dk = str(block.get("dataKey") or f"kpis_{page_id}")
    bundle = data.get(dk)
    items = block.get("items") or []
    lines: List[str] = []
    if isinstance(items, list) and items:
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("valueKey") or "KPI")
            vk = str(item.get("valueKey") or "")
            val = ""
            if isinstance(bundle, dict) and vk:
                val = bundle.get(vk, "")
            lines.append(f"  • {label}: {val}")
    elif isinstance(bundle, dict):
        for k, v in bundle.items():
            lines.append(f"  • {k}: {v}")
    return "\n".join(lines) if lines else "  (no KPI values)"


def _is_ecommerce_definition(defn: Dict[str, Any]) -> bool:
    return defn.get("appKind") == "ecommerce" or isinstance(defn.get("products"), list)


def _is_sap_similarity_definition(defn: Dict[str, Any]) -> bool:
    from app_builder.services.sap_similar_tickets import is_sap_similarity_app

    return is_sap_similarity_app(defn)


def build_sap_rag_export(defn: Dict[str, Any]) -> str:
    """PDF/text export: one ticket number + solution text per entry (no KPIs, counts, or dashboard noise)."""
    data = _data_bag(defn)
    meta = defn.get("meta") or {}
    lines: List[str] = [
        f"Application: {meta.get('appName') or 'SAP Similarity'}",
        "Format: ticket number and solution text only.",
        "",
    ]
    tickets = data.get("source_sap_tickets")
    if not isinstance(tickets, list):
        tickets = []
    rag_bag = data.get("sap_ticket_rag") if isinstance(data.get("sap_ticket_rag"), dict) else {}
    seen: set[str] = set()
    for row in tickets:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("ticket_id") or "").strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        rag = rag_bag.get(tid) if isinstance(rag_bag, dict) else {}
        solution = ""
        if isinstance(rag, dict):
            solution = str(rag.get("suggested_answer") or "").strip()
        if not solution or "step 2" in solution.lower():
            from app_builder.services.sap_community_rag import build_ticket_rag_record

            built = build_ticket_rag_record(row, force=True)
            solution = str(built.get("suggested_answer") or "")
        solution = re.sub(r"\n{3,}", "\n\n", solution).strip()
        lines.append(f"Ticket {tid}")
        lines.append(solution or "(No solution loaded — run Step 2 RAG enrich.)")
        lines.append("")
    return "\n".join(lines).strip()


def _build_sap_knowledge_chunks(defn: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Chat RAG: ticket id + solution (no KPI/table/chart export noise)."""
    data = _data_bag(defn)
    chunks: List[Tuple[str, str]] = []
    tickets = data.get("source_sap_tickets")
    rag_bag = data.get("sap_ticket_rag") if isinstance(data.get("sap_ticket_rag"), dict) else {}
    if not isinstance(tickets, list):
        return chunks
    seen: set[str] = set()
    for row in tickets:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("ticket_id") or "").strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        rag = rag_bag.get(tid) if isinstance(rag_bag, dict) else {}
        solution = ""
        if isinstance(rag, dict):
            solution = str(rag.get("suggested_answer") or "").strip()
        if not solution:
            continue
        chunks.append((f"ticket:{tid}", f"Ticket {tid}\n{solution}"))
    return chunks


def _build_ecommerce_knowledge_chunks(defn: Dict[str, Any]) -> List[Tuple[str, str]]:
    meta = defn.get("meta") or {}
    chunks: List[Tuple[str, str]] = []
    hero = defn.get("hero") or {}
    intro = (
        f"Store: {meta.get('appName', 'Online Shop')}\n"
        f"Vertical: {meta.get('domain', 'general')}\n"
        f"Tagline: {hero.get('subtitle') or meta.get('description', '')}"
    ).strip()
    chunks.append(("store", intro))

    products = defn.get("products") or []
    if isinstance(products, list) and products:
        lines = ["Product catalog (prices in INR):"]
        for p in products[:24]:
            if not isinstance(p, dict):
                continue
            name = p.get("name", "")
            price = p.get("price", "")
            stock = p.get("stock", "")
            desc = str(p.get("description") or "")[:120]
            lines.append(f"  • {name} — ₹{price}, stock {stock}. {desc}")
        chunks.append(("products", "\n".join(lines)))

    dash = defn.get("dashboard") or {}
    if isinstance(dash, dict) and dash:
        chunks.append(
            (
                "dashboard",
                "Customer dashboard (sample/demo data):\n"
                f"  • Total orders: {dash.get('total_orders')}\n"
                f"  • Pending orders: {dash.get('pending_orders')}\n"
                f"  • Total spent: ₹{dash.get('total_spent')}\n"
                f"  • Recent orders: {dash.get('recent_orders')}",
            )
        )

    pages_hint = (
        "Store pages: Home, Products catalog, Product detail, Cart, Checkout, "
        "Order confirmation, Dashboard (order history and spend KPIs)."
    )
    chunks.append(("pages", pages_hint))
    return chunks


def build_knowledge_chunks(
    defn: Dict[str, Any],
    *,
    focus_page_id: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Return (chunk_id, text) sections for retrieval."""
    if _is_ecommerce_definition(defn):
        return _build_ecommerce_knowledge_chunks(defn)

    if _is_sap_similarity_definition(defn):
        return _build_sap_knowledge_chunks(defn)

    data = _data_bag(defn)
    meta = defn.get("meta") or {}
    chunks: List[Tuple[str, str]] = []
    app_line = (
        f"Application: {meta.get('appName', 'Dashboard')}\n"
        f"Domain: {meta.get('domain', 'operations')}\n"
        f"Description: {meta.get('description', '')}"
    ).strip()
    chunks.append(("app", app_line))

    for page in defn.get("pages") or []:
        if not isinstance(page, dict):
            continue
        pid = str(page.get("id") or "")
        if focus_page_id and pid != focus_page_id:
            continue
        title = str(page.get("title") or pid)
        subtitle = str(page.get("subtitle") or "")
        body_parts = [f"Page: {title}"]
        if subtitle:
            body_parts.append(f"Purpose: {subtitle}")
        block_lines: List[str] = []
        for block in page.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "kpiRow":
                block_lines.append("KPIs:\n" + _summarize_kpi_block(block, data, pid))
            elif btype in ("lineChart", "barChart", "pieChart"):
                block_lines.append(_summarize_chart_block(block, data))
            elif btype == "dataTable":
                block_lines.append(_summarize_table_block(block, data))
        if block_lines:
            body_parts.append("\n".join(block_lines))
        chunks.append((f"page:{pid}", "\n".join(body_parts)))

    orphan_keys = [
        k
        for k in data.keys()
        if k.startswith(("kpis_", "series_", "table_", "chart_"))
        and not any(k in text for _, text in chunks)
    ]
    if orphan_keys:
        extras = []
        for k in sorted(orphan_keys)[:12]:
            val = data[k]
            if isinstance(val, dict):
                extras.append(f"{k}: " + ", ".join(f"{a}={b}" for a, b in list(val.items())[:8]))
            elif isinstance(val, list):
                extras.append(f"{k}: {len(val)} records")
        if extras:
            chunks.append(("datasets", "Additional datasets:\n" + "\n".join(extras)))
    return chunks


def retrieve_chunks(
    query: str,
    chunks: List[Tuple[str, str]],
    *,
    top_k: int = 10,
) -> List[str]:
    """Lightweight lexical retrieval (no vector DB required for dashboard-sized data)."""
    if len(chunks) <= top_k:
        return [text for _, text in chunks]
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not q_tokens:
        return [text for _, text in chunks[:top_k]]
    scored: List[Tuple[int, str]] = []
    for _cid, text in chunks:
        low = text.lower()
        score = sum(1 for t in q_tokens if t in low)
        if "kpi" in q_tokens and "kpi" in low:
            score += 2
        if any(w in q_tokens for w in ("chart", "graph", "trend", "line", "bar")) and "chart" in low:
            score += 2
        if any(w in q_tokens for w in ("product", "price", "stock", "order", "cart", "laptop")):
            if "product" in low or "catalog" in low or "₹" in text:
                score += 2
        if any(
            w in q_tokens
            for w in ("ticket", "idoc", "sap", "incident", "reconciliation", "determination")
        ):
            if "sap ticket" in low or "ticket store" in low or "RETMFG" in text:
                score += 3
        scored.append((score, text))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    picked = [t for s, t in scored if s > 0][:top_k]
    if not picked:
        picked = [text for _, text in chunks[:top_k]]
    return picked


def _is_manufacturing_definition(defn: Dict[str, Any]) -> bool:
    meta = defn.get("meta") if isinstance(defn.get("meta"), dict) else {}
    blob = " ".join(
        str(meta.get(k) or "")
        for k in ("appName", "domain", "description", "requirement")
    ).lower()
    return bool(re.search(r"manufactur|plant|vendor|procurement|warehouse|inspection", blob))


def _prompt_text(defn: Dict[str, Any]) -> str:
    meta = defn.get("meta") if isinstance(defn.get("meta"), dict) else {}
    return str(meta.get("requirement") or meta.get("description") or "").strip()


def _policies_from_prompt(defn: Dict[str, Any]) -> str:
    """Policy section follows the user prompt. Unmentioned domains are left out."""
    prompt = _prompt_text(defn)
    low = prompt.lower()
    lines = [
        "Policies",
        "These policies are taken from the prompt for this app.",
        "",
        f"Prompt: {prompt or 'Manufacturing operations for 2025 and 2026.'}",
        "",
        "- Data policy: dashboard charts, tables, and this export use the same rows, dated January 2025 through December 2026.",
    ]
    rules = (
        (
            ("safety", "stock", "inventory", "warehouse", "reorder", "sku"),
            "- Inventory policy: a SKU is below policy when quantity on hand is at or under its reorder point or safety stock.",
        ),
        (
            ("quality", "inspection", "defect", "yield", "scrap", "vendor"),
            "- Quality policy: inspection lots stay in the same lot IDs shown on the dashboard, and a lot is held when its defect rate is above the yield target in the prompt.",
        ),
        (
            ("purchase", "procurement", "spend", "po", "supplier"),
            "- Procurement policy: purchase-order value and lead time are taken from the order rows, and an order is open until its status is received or closed.",
        ),
        (
            ("sales", "revenue", "customer", "order"),
            "- Sales policy: manufacturing sales are booked order value in the same 2025–2026 window as the charts.",
        ),
        (
            ("maintenance", "mtbf", "mttr", "downtime", "oee", "work order"),
            "- Maintenance policy: downtime, MTBF, and MTTR are calculated from work-order events, not from a separate chart seed.",
        ),
    )
    matched = [text for words, text in rules if any(w in low for w in words)]
    if not matched:
        matched.append(
            "- Operating policy: each KPI and chart states a figure that can be recomputed from the rows in this document."
        )
    lines.extend(matched)
    return "\n".join(lines)


def _manufacturing_sales_section(defn: Dict[str, Any]) -> str:
    data = _data_bag(defn)
    lines = ["Manufacturing sales", ""]
    value_keys = ("order_value", "order_total", "amount", "revenue", "sales", "unit_cost")
    id_keys = ("po_number", "order_id", "inspection_id", "sku", "sales_order")
    party_keys = ("customer", "vendor", "account", "warehouse")
    date_keys = ("ordered_at", "inspected_at", "occurred_at", "date", "period")
    picked: List[str] = []
    total = 0.0
    counted = 0
    for key, rows in data.items():
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            continue
        sample = rows[0]
        if not any(k in sample for k in value_keys + id_keys):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            amount = None
            for vk in value_keys:
                val = row.get(vk)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    amount = float(val)
                    break
            ident = next((str(row.get(k)) for k in id_keys if row.get(k)), "")
            party = next((str(row.get(k)) for k in party_keys if row.get(k)), "")
            when = next((str(row.get(k))[:16] for k in date_keys if row.get(k)), "")
            if amount is None and not ident:
                continue
            counted += 1
            if amount is not None:
                total += amount
            if len(picked) < 12 and (ident or amount is not None):
                bits = [b for b in (ident, party, when, None if amount is None else f"{amount:,.2f}") if b]
                picked.append("  • " + " | ".join(bits))
    if not picked:
        lines.append("No order rows are loaded yet. Sales figures appear here once purchase or sales rows are on the dashboard.")
        return "\n".join(lines)
    lines.append(f"Orders in this export: {counted}")
    lines.append(f"Manufacturing sales value (sum of order amounts): {total:,.2f}")
    lines.append("")
    lines.extend(picked)
    return "\n".join(lines)


def build_manufacturing_rag_export(defn: Dict[str, Any]) -> str:
    """Manufacturing RAG document: Knowledge, Manufacturing sales, and prompt policies."""
    chunks = build_knowledge_chunks(defn, focus_page_id=None)
    knowledge = "\n\n".join(text for _, text in chunks) or "No dashboard knowledge is loaded."
    parts = [
        "Knowledge",
        knowledge,
        "",
        _manufacturing_sales_section(defn),
        "",
        _policies_from_prompt(defn),
    ]
    body = "\n".join(parts).strip()
    if len(body) > _MAX_CONTEXT_CHARS:
        body = body[:_MAX_CONTEXT_CHARS] + "\n…(truncated)"
    return body


def build_rag_context(
    defn: Dict[str, Any],
    query: str,
    *,
    focus_page_id: Optional[str] = None,
) -> str:
    export_query = bool(re.search(r"\b(export|download|full)\b", (query or "").lower()))
    if _is_sap_similarity_definition(defn) and export_query:
        return build_sap_rag_export(defn)
    if _is_manufacturing_definition(defn) and export_query:
        return build_manufacturing_rag_export(defn)
    chunks = build_knowledge_chunks(defn, focus_page_id=focus_page_id)
    if _is_sap_similarity_definition(defn):
        selected = [text for _, text in chunks]
    else:
        selected = retrieve_chunks(query, chunks, top_k=12)
    body = "\n\n---\n\n".join(selected)
    if len(body) > _MAX_CONTEXT_CHARS:
        body = body[:_MAX_CONTEXT_CHARS] + "\n…(truncated)"
    return body


ECOMMERCE_CHAT_SYSTEM_TEMPLATE = """You are the in-app shopping data agent for "{app_name}".
Answer questions about products, prices, stock, orders, and dashboard stats using ONLY the STORE DATA below.
Rules:
- Quote exact product names and prices (₹) from the catalog when relevant.
- For dashboard questions, use total orders, pending orders, total spent, and recent orders from the data.
- Help shoppers compare products, find items in stock, and understand their order history (demo data).
- If the question is outside the store data, say so — do not invent products or prices.
- Be friendly, concise, and use bullet points when listing multiple products.

STORE DATA:
{context}
"""

CHAT_SYSTEM_TEMPLATE = """You are the in-app analytics assistant for "{app_name}".
Answer questions about KPIs, charts, tables, trends, and operational metrics using ONLY the APPLICATION DATA below.
Topics may include supply chain, inventory, procurement, vendors, quality, plant performance, OEE, downtime, or spend — use whatever fields appear in the data.
Rules:
- Quote exact numbers from the data when available (KPI values, chart stats, table samples).
- Name the page or widget (chart title, KPI label) when referencing a metric.
- Compare trends when the data includes time series or multiple points.
- If the data does not contain enough information, say what is missing — do not invent metrics.
- Be concise, professional, and use bullet points for multi-part answers.

APPLICATION DATA:
{context}
"""
