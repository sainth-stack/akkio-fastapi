"""Styling agent — structured design tokens + production app.css from PRD/UIUX."""

from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from app_builder.services.app_generators import generate_app_css


def _extract_primary_hex(uiux: str) -> str:
    if not uiux:
        return "#4f46e5"
    match = re.search(r"Primary:\s*(#[0-9A-Fa-f]{3,8})", uiux, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(#[0-9A-Fa-f]{6})", uiux)
    return match.group(1) if match else "#4f46e5"


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
    app_css = generate_app_css(primary)
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
        app_css = generate_app_css(tokens.get("colors", {}).get("primary", primary))

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

CSS must include: .app, .card, .btn, .btn-primary, .input, .form-row, .list-item, responsive @media rules.
Match colors from the UI/UX palette exactly."""

    user_prompt = f"""Requirement:
{requirement}

PRD:
{(prd or '')[:6000]}

UI/UX:
{(uiux or '')[:6000]}

Produce the design system JSON block and complete app.css."""

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
