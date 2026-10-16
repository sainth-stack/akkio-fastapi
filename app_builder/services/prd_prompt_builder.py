"""Build PRD system/user prompts from the user requirement (prompt-faithful, page-aware)."""
from __future__ import annotations

import re
from typing import List, Tuple

from app_builder.services.app_definition_generator import (
    _normalize_prompt_text,
    _screens_from_numbered_prompt_list,
    _screens_from_requirement_phrase,
    resolve_app_display_name,
)

_GENERIC_PRD_SYSTEM = """You are a Product Requirements Document (PRD) specialist.
Write a CLEAR, prompt-faithful PRD in Markdown. The user's prompt is the single source of truth.

Rules:
- Do not add login/authentication unless the user asked for it.
- Do not switch domains (e.g. do not write "anomaly detection" if the user asked for manufacturing quality).
- Use short sentences and bullet lists; avoid long paragraphs.
- Feature names, page names, and metrics must come from the user prompt.

Structure (use these headings exactly):

# [Product Name]

## Product Overview
[2–4 sentences describing what the product is and who uses it — from the prompt only]

## Goals
- [3–5 measurable goals tied to the prompt]

## Application Pages
[One subsection per page the user requested — see page list in the user message]

## Functional Requirements
- [Actionable FR bullets; each should map to a page or cross-cutting behavior from the prompt]

## Non-Functional Requirements
- [Brief NFR: performance, usability, export/print if mentioned]

Formatting:
- Under **Application Pages**, use `### PageName` for each screen (exact names from the user).
- For each page include only the subsections that apply, using these labels:
  **Purpose:**, **KPIs:**, **Charts:**, **Filters:**, **Actions:**, **Table columns:**
- Charts: prefix with type — `Line chart:`, `Bar chart:`, `Donut chart:` (one idea per bullet).
- KPIs: one metric per bullet (e.g. First Pass Yield (FPY), Defect Rate).
- Do not duplicate the same widget on every page unless the prompt says so.
- **KPIs:** subsection ONLY on the Dashboard (or Home/Overview) page — never on other pages. Other pages use Charts, Table columns, Filters, Actions as needed.
- Do not add a Tech Stack section unless building a fullstack enterprise app."""


def _page_layout_hint(page: str) -> str:
    low = (page or "").lower()
    if re.search(r"\b(dashboard|home|overview)\b", low):
        return (
            "Layout: **KPIs** on Dashboard. If the user lists charts under **Include:** or **Charts:** "
            "(e.g. line chart for downtime trends, bar chart by category), copy those bullets here — do not omit them."
        )
    if re.search(r"\breports?\b", low):
        return "Layout: **Charts** + **Actions** (Export PDF/CSV). No **Table columns**. Omit **Filters** unless the user requested filters."
    if re.search(r"\banalytics?\b", low):
        return "Layout: **Charts** + **Table columns** with realistic field names (machine, line, shift, root cause, severity). Omit **Filters** unless the user requested filters."
    return ""


def infer_pages_from_requirement(requirement: str) -> List[str]:
    combined = _normalize_prompt_text(requirement or "")
    screens = _screens_from_numbered_prompt_list(combined)
    if len(screens) < 2:
        screens = _screens_from_requirement_phrase(combined)
    titles = [str(s.get("title") or "").strip() for s in screens if s.get("title")]
    return titles


def is_multi_page_dashboard_prompt(requirement: str) -> bool:
    req = requirement or ""
    pages = infer_pages_from_requirement(req)
    if len(pages) >= 2:
        return True
    return bool(
        re.search(
            r"\b(dashboard|sessions?|reports?|analytics)\b.*\b(dashboard|sessions?|reports?|analytics)\b",
            req,
            re.I,
        )
        or re.search(r"with\s+\d+\s+pages?", req, re.I)
    )


def _page_spec_template(page: str) -> str:
    hint = _page_layout_hint(page)
    hint_line = f"\n[{hint}]\n" if hint else "\n"
    return f"""### {page}
**Purpose:** [One sentence — what the user does on {page}; infer from the prompt, any domain]
{hint_line}
**KPIs:**
- [Metric bullets ONLY if this page shows summary numbers; otherwise omit this whole subsection]

**Charts:**
- [Line/Bar/Donut chart bullets with concrete metrics — e.g. "Line chart: OEE % vs target over time"]

**Filters:**
- [Search/date/select filters needed on this page only]

**Actions:**
- [Buttons such as Export PDF, Create record — only if relevant]

**Table columns:**
- [Column bullets ONLY if this page is a list, log, or catalog]
"""


def build_prd_prompts(
    requirement: str,
    *,
    fullstack_addendum: str = "",
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for PRD generation."""
    product_name = resolve_app_display_name(requirement=requirement)
    pages = infer_pages_from_requirement(requirement)

    maintenance = bool(re.search(r"maintenance|mtbf|mttr|work order|downtime", requirement or "", re.I))
    kpi_defs = ""
    if maintenance:
        kpi_defs = """
## KPI Definitions
- **Total Maintenance Work Orders:** Count of open + closed work orders in the selected period.
- **MTBF (Mean Time Between Failures):** Average operating time between unplanned failures (hours).
- **MTTR (Mean Time to Repair):** Average time from failure detection to restored operation (hours).

## Maintenance status workflow
- **Scheduled** → **In progress** → **Completed** (preventive/corrective)
- **Breakdown reported** → **Triaged** → **Repair in progress** → **Resolved** / **Deferred**
- Use green for operational/resolved, amber for scheduled/warning, red for critical/breakdown.
"""

    if is_multi_page_dashboard_prompt(requirement):
        page_block = "\n".join(_page_spec_template(p) for p in pages) if pages else _page_spec_template("Dashboard")
        system_prompt = f"""You are a senior product manager writing a PRD for a multi-page SaaS web application.

Product name (short brand — 1–2 words for the app shell): {product_name}

The user prompt defines pages, KPIs, charts, tables, and workflows. You MUST:
1. Use the exact application page names: {", ".join(pages) if pages else "as stated in the prompt"}.
2. Expand each page with concrete KPIs, charts, filters, and table columns implied by the prompt (manufacturing quality, operations, etc. when applicable).
3. NEVER use document section titles (Product Overview, Business Requirements, Functional Requirements) as application page names.
4. NEVER replace the product with a different concept (e.g. generic anomaly platform).

Required Markdown outline:

# {product_name}

## Product Overview

## Goals

## Application Pages
{page_block}
{kpi_defs}
## Functional Requirements

## Non-Functional Requirements

Keep **Process flows / use cases** to at most 3 short bullets under Goals OR omit them.
Chart bullets must start with Line chart / Bar chart / Donut chart.
"""
        if fullstack_addendum:
            system_prompt += "\n" + fullstack_addendum
    else:
        system_prompt = _GENERIC_PRD_SYSTEM
        if fullstack_addendum:
            system_prompt += "\n" + fullstack_addendum

    page_line = ", ".join(pages) if pages else "(derive from prompt)"
    user_prompt = f"""Write the PRD for this product.

--- USER PROMPT (source of truth) ---
{requirement.strip()}
--- END PROMPT ---

Product name: {product_name}
Application pages to document under ## Application Pages: {page_line}

Follow the structure in your instructions. Be specific and implementation-ready for UI generation (KPI cards, charts, tables, filters)."""

    return system_prompt, user_prompt
