"""Validate every navigated screen has visible widgets; fill gaps from PRD + domain defaults."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_VISIBLE_TYPES = frozenset({"kpiRow", "lineChart", "barChart", "pieChart", "dataTable"})


def _visible_widgets(blocks: List[Any]) -> List[Dict[str, Any]]:
    return [
        b
        for b in blocks or []
        if isinstance(b, dict) and str(b.get("type") or "") in _VISIBLE_TYPES
    ]


def _synthetic_section(page_id: str, title: str, subtitle: str, *, inventory: bool) -> str:
    if inventory:
        return f"""### {title}
{subtitle or title}

#### Charts
- Stock levels by warehouse (bar chart)
- Units on hand trend over time (line chart)

#### Table columns
- sku
- product_name
- warehouse
- quantity_on_hand
- reorder_point
- status
"""
    if re.search(r"reports?", f"{page_id} {title}", re.I):
        return f"""### {title}
{subtitle or title}

#### Charts
- Summary trend for leadership review (line chart)
- Breakdown by category (bar chart)
"""
    if re.search(r"analytics|forecast", f"{page_id} {title}", re.I):
        return f"""### {title}
{subtitle or title}

#### Charts
- Trend analysis (line chart)
- Comparison by dimension (bar chart)
"""
    return f"""### {title}
{subtitle or title}

#### Charts
- {title} trend (line chart)
- {title} breakdown (bar chart)

#### Table columns
- id
- name
- status
- updated_at
"""


def validate_all_screens(
    defn: Dict[str, Any],
    *,
    prd: str = "",
    requirement: str = "",
) -> Dict[str, Any]:
    """
    Ensure each navigation target (except predict) has ≥2 visible blocks.
    Safe to run on every /definition request — only fills gaps, does not wipe good layouts.
    """
    from app_builder.schemas.app_definition import AppDefinition, PageDef
    from app_builder.services.page_layout_intent import resolve_dashboard_page_id
    from app_builder.services.prd_driven_layout import blocks_for_page
    from app_builder.services.prd_structure_parser import page_prompt_context
    from app_builder.services.source_dataset_planner import _ctx

    try:
        model = AppDefinition.model_validate(defn)
    except Exception as exc:
        logger.warning("[validate_screens] skip invalid definition: %s", exc)
        return defn

    ctx = _ctx(defn, requirement, prd)
    inventory_app = bool(re.search(r"inventory|stock|sku|warehouse", ctx))

    pages_by_id = {p.id: p for p in model.pages}
    nav_ids = [
        n.pageId for n in model.navigation if n.pageId not in ("predict", "observability")
    ]
    dash_id = resolve_dashboard_page_id([(p.id, p.title) for p in model.pages], nav_ids)

    changed = False
    new_pages: List[PageDef] = []

    for page in model.pages:
        if page.id in ("predict", "observability"):
            new_pages.append(page)
            continue
        if page.id not in nav_ids and page.id != dash_id:
            new_pages.append(page)
            continue

        visible = _visible_widgets(list(page.blocks or []))
        if len(visible) >= 2:
            new_pages.append(page)
            continue

        section = page_prompt_context(prd, requirement, page.title, page.id)
        if not section.strip():
            section = _synthetic_section(
                page.id,
                page.title,
                page.subtitle or "",
                inventory=inventory_app
                or bool(re.search(r"inventory|stock|sku", f"{page.id} {page.title}", re.I)),
            )

        rebuilt = blocks_for_page(
            page.id,
            page.title,
            section,
            dashboard_page_id=dash_id,
        )
        if len(_visible_widgets(rebuilt)) < 2:
            from app_builder.services.prd_driven_layout import default_page_chart_blocks

            rebuilt = list(rebuilt or [])
            rebuilt.extend(
                default_page_chart_blocks(
                    page.id,
                    page.title,
                    inventory=inventory_app,
                    cap=2,
                )
            )
            if "dataTable" not in {b.get("type") for b in rebuilt}:
                from app_builder.services.source_dataset_planner import _inventory_columns

                cols = _inventory_columns() if inventory_app else [
                    {"field": "id", "header": "ID"},
                    {"field": "name", "header": "Name"},
                    {"field": "status", "header": "Status"},
                ]
                rebuilt.append(
                    {
                        "type": "dataTable",
                        "title": f"{page.title} records",
                        "dataKey": f"table_{page.id}",
                        "columns": cols[:8],
                        "searchable": True,
                        "detailView": True,
                    }
                )

        logger.info(
            "[validate_screens] filled page=%s visible_before=%s visible_after=%s",
            page.id,
            len(visible),
            len(_visible_widgets(rebuilt)),
        )
        new_pages.append(
            PageDef(id=page.id, title=page.title, subtitle=page.subtitle, blocks=rebuilt)
        )
        changed = True

    if changed:
        model.pages = new_pages
        return model.model_dump(mode="json")
    return defn
