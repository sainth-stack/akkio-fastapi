"""Production-style datasets — unique per page and block."""
from __future__ import annotations

from typing import Any, Dict, List

from app_builder.schemas.app_definition import AppDefinition
from app_builder.services.page_mock_catalog import (
    CHART_POINTS,
    MIN_TABLE_ROWS,
    enrich_kpi_object,
    generate_chart_rows,
    generate_pie_rows,
    generate_table_rows,
    infer_page_profile,
    should_replace_table,
)


def _sync_kpi_items(block: Dict[str, Any], bundle: Dict[str, str], page_id: str) -> None:
    from app_builder.services.block_normalizer import _KPI_COLORS, _humanize

    data_key = str(block.get("dataKey") or f"kpis_{page_id}")
    block["dataKey"] = data_key
    block["items"] = [
        {
            "label": _humanize(k),
            "valueKey": k,
            "color": _KPI_COLORS[i % len(_KPI_COLORS)],
        }
        for i, k in enumerate(bundle.keys())
    ]


def enrich_mvp_mock(defn: AppDefinition) -> AppDefinition:
    """Replace placeholders with page-appropriate, non-duplicated production data."""
    mock: Dict[str, Any] = dict(defn.mock or {})
    meta_desc = defn.meta.description or ""
    domain = defn.meta.domain or ""

    for page in defn.pages:
        profile = infer_page_profile(
            page.id,
            page.title,
            page.subtitle or "",
            meta_desc,
            domain,
        )

        for block in page.blocks:
            btype = block.get("type")

            if btype == "kpiRow":
                data_key = str(block.get("dataKey") or f"kpis_{page.id}")
                existing = mock.get(data_key)
                if not isinstance(existing, dict):
                    existing = {}
                bundle = enrich_kpi_object(existing, profile, page.id)
                mock[data_key] = bundle
                mock[f"kpis_{page.id}"] = bundle
                _sync_kpi_items(block, bundle, page.id)

            elif btype in ("lineChart", "barChart"):
                dk_s = str(block.get("dataKey") or "")
                if not dk_s:
                    continue
                series = block.get("series") or []
                series_keys = [
                    str(s.get("dataKey")) for s in series if isinstance(s, dict) and s.get("dataKey")
                ]
                existing = mock.get(dk_s)
                needs = (
                    not existing
                    or not isinstance(existing, list)
                    or len(existing) < CHART_POINTS - 2
                    or _looks_placeholder_series(existing)
                    or _series_keys_mismatch(existing, series_keys)
                )
                if needs:
                    if not series_keys and isinstance(existing, list) and existing and isinstance(existing[0], dict):
                        x_key = str(block.get("xAxisKey") or "period")
                        if x_key not in existing[0]:
                            for cand in ("period", "label", "name", "week"):
                                if cand in existing[0]:
                                    x_key = cand
                                    break
                        series_keys = [k for k in existing[0].keys() if k != x_key]
                    if not series_keys:
                        series_keys = (
                            ["load_mva", "target"]
                            if profile == "transformer_monitoring"
                            else ["output", "target"]
                        )
                    rows, x_key = generate_chart_rows(
                        dk_s, series_keys, profile, btype, page_id=page.id
                    )
                    block["xAxisKey"] = x_key
                    if not block.get("series"):
                        block["series"] = [
                            {"dataKey": k, "name": k.replace("_", " ").title()} for k in series_keys
                        ]
                    mock[dk_s] = rows

            elif btype == "pieChart":
                dk_s = str(block.get("dataKey") or "")
                if not dk_s:
                    continue
                existing = mock.get(dk_s)
                if not isinstance(existing, list) or len(existing) < 3:
                    mock[dk_s] = generate_pie_rows(dk_s, profile)

            elif btype == "dataTable":
                dk_s = str(block.get("dataKey") or "")
                if not dk_s:
                    continue
                cols = block.get("columns") or []
                existing = mock.get(dk_s)
                if should_replace_table(existing if isinstance(existing, list) else []):
                    mock[dk_s] = generate_table_rows(
                        cols,
                        profile=profile,
                        table_key=dk_s,
                        page_title=page.title,
                    )
                elif isinstance(existing, list) and len(existing) < MIN_TABLE_ROWS:
                    mock[dk_s] = generate_table_rows(
                        cols,
                        profile=profile,
                        table_key=dk_s,
                        page_title=page.title,
                        row_count=MIN_TABLE_ROWS,
                    )

    defn.mock = mock
    return defn


def _series_keys_mismatch(rows: List[Any], series_keys: List[str]) -> bool:
    if not rows or not isinstance(rows[0], dict) or not series_keys:
        return False
    return not all(k in rows[0] for k in series_keys)


def _looks_placeholder_series(rows: List[Any]) -> bool:
    if not rows or not isinstance(rows[0], dict):
        return True
    r0 = rows[0]
    vals = [v for v in r0.values() if isinstance(v, (int, float))]
    return len(vals) > 0 and all(v in (70, 73, 76) for v in vals[:3])
