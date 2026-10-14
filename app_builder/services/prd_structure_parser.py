"""Extract page-level requirements from PRD / prompt text (domain-agnostic)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def _normalize_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _strip_heading_prefix(raw: str) -> str:
    t = (raw or "").strip()
    t = re.sub(r"^#{1,6}\s+", "", t)
    t = re.sub(r"^\d+[\).\s]+", "", t)
    t = re.sub(r"^\*+\s*", "", t)
    t = re.sub(r"\s*\*+$", "", t)
    return t.strip()


def _title_matches_page(heading: str, page_title: str, page_id: str) -> bool:
    nt = _normalize_title(_strip_heading_prefix(heading))
    want = {
        _normalize_title(page_title),
        _normalize_title(page_id.replace("_", " ")),
        _normalize_title(page_id),
    }
    want.discard("")
    if nt in want:
        return True
    for w in want:
        if len(w) < 3:
            continue
        if w == nt or w in nt or nt in w:
            return True
    return False


def split_prd_by_sections(text: str) -> List[Tuple[str, str]]:
    """Return (heading, body) pairs from markdown-style PRD."""
    if not text:
        return []
    sections: List[Tuple[str, str]] = []
    current_title = "Overview"
    current_lines: List[str] = []
    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.+)$", line.strip())
        if m:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = _strip_heading_prefix(m.group(1).strip())
            current_lines = []
        else:
            m2 = re.match(r"^###\s*\d*\.?\s*(.+)$", line.strip())
            if m2:
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = _strip_heading_prefix(m2.group(1).strip())
                current_lines = []
            else:
                current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def _slice_section_from_markdown(prd: str, page_title: str, page_id: str) -> str:
    """Find a page section by scanning markdown headings in raw PRD text."""
    if not prd:
        return ""
    lines = prd.splitlines()
    start: Optional[int] = None
    heading = ""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not re.match(r"^#{1,4}\s+", stripped):
            continue
        raw = _strip_heading_prefix(re.sub(r"^#{1,4}\s+", "", stripped))
        if _title_matches_page(raw, page_title, page_id):
            start = i + 1
            heading = raw
            break
    if start is None:
        return ""
    body_lines: List[str] = []
    for line in lines[start:]:
        if re.match(r"^#{1,4}\s+", line.strip()):
            break
        body_lines.append(line)
    body = "\n".join(body_lines).strip()
    if not body:
        return ""
    return f"## {heading}\n{body}"


def section_for_page(prd: str, page_title: str, page_id: str) -> str:
    """Best-matching PRD section body for a page."""
    sliced = _slice_section_from_markdown(prd, page_title, page_id)
    if sliced:
        return sliced

    sections = split_prd_by_sections(prd)
    for title, body in sections:
        if _title_matches_page(title, page_title, page_id):
            return f"## {title}\n{body}"
    return ""


def build_planning_context(
    *,
    requirement: str = "",
    prd: str = "",
    uiux: str = "",
    plan_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble full context for LLM (minimal truncation)."""
    plan_json = plan_json or {}
    req = requirement or ""
    prd_full = (prd or "").strip() or req
    combined = "\n\n".join(
        x for x in [req, prd_full, uiux or "", _architecture_excerpt(plan_json)] if x.strip()
    )
    return {
        "requirement": req,
        "prd": prd_full[:48000],
        "uiux": (uiux or "")[:24000],
        "combined": combined[:64000],
        "prd_sections": split_prd_by_sections(prd_full),
        "plan_json": plan_json,
    }


def _architecture_excerpt(plan_json: Dict[str, Any]) -> str:
    arch = plan_json.get("architecture")
    if not isinstance(arch, dict):
        return ""
    import json

    return json.dumps(arch, indent=2)[:16000]
