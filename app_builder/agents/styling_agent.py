"""Styling agent — structured design tokens + production app.css from PRD/UIUX."""

from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app_builder.services.design_system_css import build_css_from_payload, extract_primary_hex


def _extract_primary_hex(uiux: str) -> str:
    return extract_primary_hex(uiux)


def _parse_style_response(full_response: str, uiux: str) -> Dict[str, Any]:
    """Parse LLM JSON block; fall back to heuristic tokens + generated CSS."""
    primary = _extract_primary_hex(uiux)
    tokens: Dict[str, Any] = {
        "colors": {
            "primary": primary,
            "background": "#f8fafc",
            "surface": "#ffffff",
            "text": "#0f172a",
            "muted": "#64748b",
            "border": "#e2e8f0",
            "danger": "#ef4444",
        },
        "typography": {"fontFamily": "Inter, system-ui, sans-serif"},
        "layout": {"shell": "topbar+content", "maxContentWidth": "80rem"},
    }
    app_css = build_css_from_payload({"design_tokens": tokens}, uiux=uiux)
    summary_md = full_response.strip()

    json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", full_response)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed.get("design_tokens"), dict):
                tokens = parsed["design_tokens"]
            elif isinstance(parsed, dict) and parsed.get("colors"):
                tokens = parsed
            if isinstance(parsed.get("app_css"), str) and len(parsed["app_css"]) > 80:
                app_css = parsed["app_css"]
            if isinstance(parsed.get("summary_md"), str):
                summary_md = parsed["summary_md"]
        except json.JSONDecodeError:
            pass

    if len(app_css.strip().split("\n")) < 40:
        app_css = build_css_from_payload({"design_tokens": tokens}, uiux=uiux)

    return {
        "design_tokens": tokens,
        "app_css": app_css,
        "summary_md": summary_md or f"Design system with primary {tokens['colors']['primary']}.",
    }


async def stream_styling_generation(
    requirement: str,
    prd: str,
    uiux: str,
    llm,
    builder_kind: Optional[str] = None,
    track: Optional[str] = None,
    prd_json: Optional[Dict] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    system_prompt = """You are a senior product designer building SaaS-grade design systems.
Given PRD and UI/UX specs, output:
1. A JSON code block with design_tokens (colors, typography, spacing, layout) and optional app_css (plain CSS with :root variables).
2. Use plain CSS variables — NOT Tailwind utilities in generated CSS.

Required JSON shape:
```json
{
  "design_tokens": {
    "colors": {
      "primary": "#hex",
      "background": "#hex",
      "surface": "#hex",
      "text": "#hex",
      "muted": "#hex",
      "border": "#hex",
      "danger": "#hex"
    },
    "typography": { "fontFamily": "Inter, system-ui, sans-serif" },
    "layout": { "shell": "sidebar+topbar|topbar+content", "maxContentWidth": "80rem" }
  },
  "app_css": "/* full frontend/src/styles/app.css content with :root and component classes */",
  "summary_md": "# Design System\\n\\nBrief human-readable summary with color swatches as hex codes."
}
```

CSS must include: .app, .app-container, .app-header, .card, .btn, .btn-primary, .input, .form-row, .list-item, body background, :root color variables, responsive @media rules.
Match colors from the UI/UX palette exactly. The platform injects the final app.css — focus on accurate design_tokens colors."""

    user_prompt = f"""Requirement:
{requirement}

PRD:
{(prd or '')[:6000]}

UI/UX:
{(uiux or '')[:6000]}

Produce the design system JSON block and complete app.css."""

    from app_builder.services.fullstack_stack import is_fullstack
    if is_fullstack(builder_kind):
        system_prompt += (
            "\nAlso emit MUI palette fields (primary.main, background.default, text.primary) "
            "matching the prompt colors. This theme is applied via Material UI createTheme."
        )
        user_prompt += "\nExtract hex colors from the requirement/UI-UX exactly. Enterprise light theme unless specified."

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    full_response = ""
    yield {"event": "style_start", "message": "Generating design system..."}

    async for chunk in llm.astream(messages):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        if content:
            full_response += content
            yield {"event": "style_chunk", "data": content}

    result = _parse_style_response(full_response, uiux)
    yield {"event": "style_complete", "data": result}

    # ── Frontend-only track: extract structured DesignTokenSchema JSON ─────
    if track == "frontend_only":
        try:
            yield {"event": "style_json_start", "message": "Extracting structured design tokens…"}
            style_json_data = await _emit_style_json(requirement, full_response, result, llm)
            if style_json_data:
                yield {"event": "style_json", "data": style_json_data}
            else:
                yield {"event": "style_json_error", "message": "Structured token extraction failed"}
        except Exception as _sje:
            yield {"event": "style_json_error", "message": str(_sje)}
    # ───────────────────────────────────────────────────────────────────────


# =============================================================================
# Frontend-only track: structured DesignTokenSchema extraction
# =============================================================================

_STYLE_JSON_SYSTEM = """\
You are a design-systems engineer. Extract a DesignTokenSchema JSON from the style document.
Return ONLY valid JSON — no markdown, no explanation — matching this schema:

{
  "light": {
    "primary": "#hex",
    "secondary": "#hex",
    "background": "#hex",
    "surface": "#hex",
    "text_primary": "#hex",
    "text_secondary": "#hex",
    "border": "#hex",
    "error": "#hex",
    "warning": "#hex",
    "success": "#hex"
  },
  "dark": {
    "primary": "#hex",
    "secondary": "#hex",
    "background": "#hex",
    "surface": "#hex",
    "text_primary": "#hex",
    "text_secondary": "#hex",
    "border": "#hex",
    "error": "#hex",
    "warning": "#hex",
    "success": "#hex"
  },
  "font_family": "Inter, sans-serif",
  "border_radius": "8px",
  "chart_palette": ["#hex1", "#hex2", "#hex3", "#hex4", "#hex5"]
}

Rules:
- All hex values must be valid CSS hex colors (#rrggbb)
- Dark palette must genuinely differ from light (dark bg ~#0d1117, surfaces ~#161b22)
- chart_palette: 5 distinct accessible colors
- border_radius: single CSS value e.g. "6px" or "8px"
- font_family: full CSS font stack"""


async def _emit_style_json(
    requirement: str,
    full_response: str,
    parsed_result: Dict,
    llm: Any,
) -> Optional[Dict]:
    from app_builder.agents.structured_output import extract_validated
    from app_builder.schemas.frontend_plan import DesignTokenSchema

    user_prompt = (
        f"Requirement: {requirement}\n\n"
        f"Style document:\n\n{full_response}"
    )

    model = await extract_validated(
        llm=llm,
        system_prompt=_STYLE_JSON_SYSTEM,
        user_prompt=user_prompt,
        schema=DesignTokenSchema,
        max_retries=2,
        context_label="style_json",
    )
    if model is not None:
        return model.dict()
    return None
