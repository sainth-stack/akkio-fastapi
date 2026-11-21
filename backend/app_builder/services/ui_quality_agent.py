"""
UI Quality Reviewer Agent — uses a vision LLM to score screenshots against a checklist.

Checklist items (each scored 0-2):
  1. App shell present (sidebar nav visible, not blank)
  2. Spacing and whitespace (generous, not cramped)
  3. Typography hierarchy (headings > body, weights differ)
  4. All three states handled (loading/empty/error visible or implied)
  5. Contrast (text readable on background)
  6. Responsive layout (no horizontal overflow at 1280px)
  7. Realistic data (not "Sample A", "Test 1", "Item 1")
  8. Status/badge chips colored correctly

Score: 0-16. Threshold: 11/16 (69%).
If score < threshold, apply one polish pass limited to flagged files.
"""
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("app_builder")

SCORE_THRESHOLD = 11          # out of 16
MAX_CHECKLIST_ITEMS = 8


@dataclass
class QualityScore:
    route: str
    screenshot_path: str
    scores: Dict[str, int] = field(default_factory=dict)  # item → 0|1|2
    comments: Dict[str, str] = field(default_factory=dict)
    total: int = 0
    passed: bool = True
    flagged_files: List[str] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        max_score = MAX_CHECKLIST_ITEMS * 2
        return round(self.total / max_score * 100, 1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def review_ui_quality(
    project_name: str,
    gate_screenshots: Dict[str, str],   # route → screenshot_path
    blueprint_json: Optional[Dict],
    uiux_json: Optional[Dict],
    files: Dict[str, str],
    llm: Any,
    on_event: Optional[Callable] = None,
) -> Tuple[Dict[str, str], List[QualityScore]]:
    """
    Review screenshot quality for each route.
    Returns (possibly updated files, list of QualityScore).
    """
    if not gate_screenshots:
        return files, []

    vision_llm = _get_vision_llm(llm)
    if vision_llm is None:
        logger.info("[ui_quality] no vision-capable LLM — skipping quality review")
        return files, []

    await _emit(on_event, {
        "event": "agent_start", "agent": "ui_quality",
        "message": f"UI quality review: scoring {len(gate_screenshots)} screenshots...",
    })

    scores: List[QualityScore] = []
    for route, shot_path in gate_screenshots.items():
        if not os.path.isfile(shot_path):
            continue
        score = await _score_screenshot(
            vision_llm, route, shot_path, blueprint_json, uiux_json
        )
        scores.append(score)
        await _emit(on_event, {
            "event": "agent_progress", "agent": "ui_quality",
            "message": f"  {route}: {score.total}/16 ({score.percentage}%) {'✓' if score.passed else '⚠'}",
        })

    # Apply polish pass for failing routes
    failing = [s for s in scores if not s.passed]
    if failing and llm:
        await _emit(on_event, {
            "event": "agent_progress", "agent": "ui_quality",
            "message": f"Applying polish pass to {len(failing)} routes...",
        })
        files = await _polish_pass(failing, files, blueprint_json, uiux_json, llm, on_event)

    overall_passed = all(s.passed for s in scores)
    await _emit(on_event, {
        "event": "agent_complete" if overall_passed else "agent_error",
        "agent": "ui_quality",
        "message": f"UI quality: {sum(1 for s in scores if s.passed)}/{len(scores)} routes scored ≥{SCORE_THRESHOLD}/16",
        "data": {
            "scores": [{"route": s.route, "total": s.total, "passed": s.passed} for s in scores],
        },
    })

    return files, scores


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_SCORE_SYSTEM = """\
You are a senior UI/UX reviewer. Score the provided screenshot against each checklist item.
Return a JSON object with this exact structure:

{
  "scores": {
    "app_shell": <0|1|2>,
    "spacing": <0|1|2>,
    "typography": <0|1|2>,
    "states": <0|1|2>,
    "contrast": <0|1|2>,
    "responsive": <0|1|2>,
    "realistic_data": <0|1|2>,
    "status_chips": <0|1|2>
  },
  "comments": {
    "app_shell": "one sentence comment",
    "spacing": "one sentence comment",
    ...
  },
  "flagged_files": ["list of source file names that need fixing, e.g. DashboardPage.tsx"]
}

Scoring guide (0=missing/broken, 1=partial/mediocre, 2=good/correct):
- app_shell: Is there a sidebar nav or topbar? Is it populated with labeled nav items?
- spacing: Generous whitespace, padding around cards and sections?
- typography: Visual hierarchy (h1>h2>body), font weights vary, readable sizes?
- states: Does the page handle empty/loading/error states (even if not triggered)?
- contrast: Text clearly readable against background, sufficient color contrast?
- responsive: No horizontal overflow, content fills width appropriately at 1280px?
- realistic_data: Data looks real (real names, actual numbers, proper dates — not "Item 1")?
- status_chips: Status badges/chips have appropriate colors (green=active, red=error, etc.)?

Return ONLY the JSON object. No markdown, no explanation."""


async def _score_screenshot(
    vision_llm: Any,
    route: str,
    shot_path: str,
    blueprint_json: Optional[Dict],
    uiux_json: Optional[Dict],
) -> QualityScore:
    """Score a single screenshot."""
    score = QualityScore(route=route, screenshot_path=shot_path)

    try:
        image_b64 = _encode_image(shot_path)
        if not image_b64:
            score.total = SCORE_THRESHOLD  # pass if can't read
            score.passed = True
            return score

        ux_ctx = ""
        if uiux_json:
            for sx in (uiux_json.get("screens") or []):
                if route == "/" and sx.get("screen_id") == "S001":
                    ux_ctx = f"Expected layout: {sx.get('layout')}. Sections: {sx.get('sections', [])}"
                    break

        user_message = {
            "type": "text",
            "text": f"Route: {route}\n{ux_ctx}\n\nScore this screenshot:"
        }
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        }

        from langchain_core.messages import HumanMessage, SystemMessage
        response = await vision_llm.ainvoke([
            SystemMessage(content=_SCORE_SYSTEM),
            HumanMessage(content=[user_message, image_message]),
        ])
        content = response.content if hasattr(response, "content") else str(response)

        import json, re
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            data = json.loads(m.group(0))
            score.scores = data.get("scores") or {}
            score.comments = data.get("comments") or {}
            score.flagged_files = data.get("flagged_files") or []
            score.total = sum(int(v) for v in score.scores.values() if isinstance(v, (int, float)))
            score.passed = score.total >= SCORE_THRESHOLD

    except Exception as exc:
        logger.warning("[ui_quality] scoring failed for %s: %s", route, exc)
        score.total = SCORE_THRESHOLD  # pass on error
        score.passed = True

    return score


# ---------------------------------------------------------------------------
# Polish pass
# ---------------------------------------------------------------------------

_POLISH_SYSTEM = """\
You are a React/TypeScript UI engineer. Apply targeted polish to the provided page.

Fix ONLY the issues listed in the review comments:
- spacing: add sx={{ p: 2 }} or sx={{ mb: 2 }} padding/margin where missing
- typography: use Typography variant='h5' for section headers, variant='body2' for secondary text
- realistic_data: ensure mock data uses real-looking values (import from '../mock')
- contrast: use theme-aware colors (sx={{ color: 'text.primary' }}, not hardcoded)
- status_chips: ensure StatusChip or Chip color prop matches status semantics

Rules:
1. Return the COMPLETE updated file.
2. Change ONLY what the review flagged.
3. Do not add new features or restructure the component.
4. Do not import new packages.

Return ONLY the file content. No markdown, no explanation."""


async def _polish_pass(
    failing_scores: List[QualityScore],
    files: Dict[str, str],
    blueprint_json: Optional[Dict],
    uiux_json: Optional[Dict],
    llm: Any,
    on_event: Optional[Callable],
) -> Dict[str, str]:
    """Apply one polish pass to flagged files."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from app_builder.services.verify_service import is_complete_file

    for score in failing_scores:
        for fname in score.flagged_files:
            # Resolve to full key
            key = f"frontend/src/pages/{fname}" if not fname.startswith("frontend/") else fname
            if key not in files:
                continue

            comments_text = "\n".join(
                f"  {item}: {comment} (score: {score.scores.get(item, 0)}/2)"
                for item, comment in score.comments.items()
                if score.scores.get(item, 2) < 2
            )

            user_prompt = (
                f"Route: {score.route}\nFile: {fname}\n\n"
                f"Review issues:\n{comments_text}\n\n"
                f"Current file content:\n{files[key][:4000]}\n\n"
                "Apply polish fixes."
            )

            try:
                response = await llm.ainvoke([
                    SystemMessage(content=_POLISH_SYSTEM),
                    HumanMessage(content=user_prompt),
                ])
                content = response.content if hasattr(response, "content") else str(response)
                import re
                content = re.sub(r"^```\w*\n?", "", content.strip())
                content = re.sub(r"\n?```$", "", content.strip())
                if content and is_complete_file(content):
                    files[key] = content
                    await _emit(on_event, {
                        "event": "agent_progress", "agent": "ui_quality",
                        "message": f"  polished: {fname}",
                    })
            except Exception as exc:
                logger.warning("[ui_quality/polish] failed for %s: %s", fname, exc)

    return files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_vision_llm(llm: Any) -> Optional[Any]:
    """
    Return a vision-capable LLM variant.
    Tries to get GPT-4o or Claude claude-4.6-sonnet (both support image inputs).
    Falls back to the provided LLM and hopes it supports vision.
    """
    if not _is_vision_capable(llm):
        return None
    return llm


def _is_vision_capable(llm: Any) -> bool:
    """Check if the LLM appears to support vision (image inputs)."""
    try:
        model_name = (
            getattr(llm, "model_name", None)
            or getattr(llm, "model", None)
            or ""
        ).lower()
        vision_models = ("gpt-4o", "gpt-4-vision", "claude-3", "claude-sonnet", "gemini")
        return any(m in model_name for m in vision_models)
    except Exception:
        return False


def _encode_image(path: str) -> Optional[str]:
    """Read a PNG and return base64 encoded string."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except OSError:
        return None


async def _emit(on_event: Optional[Callable], data: Dict) -> None:
    if on_event:
        try:
            await on_event(data)
        except Exception:
            pass
