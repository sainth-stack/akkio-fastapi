"""
UIUX Agent — streams UI/UX design generation.

For the frontend-only track this module also emits a validated UXPlan JSON
(event: "uiux_json") after the markdown stream finishes.
"""
from typing import List, Dict, Any, AsyncGenerator, Optional
import json
from langchain_core.messages import SystemMessage, HumanMessage

async def stream_uiux_generation(
    requirement: str,
    prd: str,
    llm,
    builder_kind: Optional[str] = None,
    track: Optional[str] = None,
    prd_json: Optional[Dict] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streams UI/UX design generation.
    """
    system_prompt = """You are a World-Class UI/UX Designer for Enterprise SaaS Applications.
Your task is to design a PREMIUM, MODERN, and HIGHLY POLISHED interface based on the PRD.
The goal is to wow the user with a "Silicon Valley" standard design (think Stripe, Linear, Vercel).

Do NOT describe a login page or authentication flow. Focus on the actual app functionality the user described.
Each screen description should explain what data is shown and what the user can do — directly related to the app's purpose.

**DESIGN AESTHETICS:**
1. **Premium & Clean**: Use generous whitespace, subtle borders, and soft shadows. Avoid "box-in-a-box" layouts.
2. **Modern Typography**: Use clean sans-serif fonts (Inter, SF Pro). clear hierarchy with weights (Medium/Semibold for headings, Regular for body).
3. **Color Palette**:
   - **Neutrals**: Use Slate/Zinc/Gray (50-950) for a slick backbone.
   - **Primary**: Use a VIBRANT, saturated primary color (e.g., Indigo-600, Violet-600, Emerald-600) for actions.
   - **Backgrounds**: Use extremely light gray (slate-50) or pure white for surfaces.
4. **Components**:
   - **Cards**: White bg, thin border (gray-200), subtle shadow-sm.
   - **Buttons**: Rounded-md or rounded-lg, varying weights (Primary vs Ghost).
   - **Inputs**: Clean borders, ring-focus states.

Output Format:

# UI/UX Design

## Color Palette
- Primary: [Hex Code]
- Secondary: [Hex Code]
- Background: [Hex Code]
- Text: [Hex Code]
- Muted/Gray: [Hex Code]

## Typography
- Headings: [Font Family, e.g., Inter, Plus Jakarta Sans]
- Body: [Font Family]

## Layout Structure
- [Detailed description of the layout shell, e.g., "Fixed top header, centered main content (max ~960px), card-based sections"]

## Key Screens
### 1. [Screen Name]
- **Layout**: [Grid/Flex structure]
- **Components**: 
  - [Component Name]: [Visual style, e.g., "Card with icon header and metric value"]
- **Interactions**: [Hover states, transitions]

(Repeat for 3-4 key screens)

## Components Styling Rules
- **Buttons**: use class `btn btn-primary` for primary actions, `btn` for secondary.
- **Cards**: use class `card` on white panels with subtle shadow.
- **Inputs**: use class `input` with clear focus states.

**CSS CLASS CONTRACT (codegen uses plain CSS, NOT Tailwind):**
- Layout: `app`, `app-container`, `app-header`, `app-title`, `app-subtitle`
- Forms: `form-row`, `input`, `btn`, `btn-primary`
- Lists: `list`, `list-item`
- Provide hex color codes in the Color Palette section — these drive the injected theme.
"""

    user_prompt = f"""Create a UI/UX Design for:

PRD Summary:
{prd}

Requirement:
{requirement}
"""

    from app_builder.services.fullstack_stack import is_fullstack, uiux_system_addendum
    if is_fullstack(builder_kind):
        system_prompt = system_prompt + "\n" + uiux_system_addendum()
        user_prompt += (
            "\n\nMap every module to a Material UI screen. Extract colors from the prompt. "
            "Include reusable components and empty/loading/error states."
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    full_response = ""
    yield {"event": "uiux_start", "message": "Generating UI/UX Design..."}
    
    async for chunk in llm.astream(messages):
        if hasattr(chunk, 'content'):
            content = chunk.content
            full_response += content
            yield {
                "event": "uiux_chunk",
                "data": content
            }
            
    yield {
        "event": "uiux_complete",
        "data": full_response
    }

    # ── Frontend-only track: extract structured UXPlan JSON ────────────────
    if track == "frontend_only":
        try:
            yield {"event": "uiux_json_start", "message": "Extracting structured UX plan…"}
            ux_json_data = await _emit_uiux_json(requirement, full_response, prd_json, llm)
            if ux_json_data:
                yield {"event": "uiux_json", "data": ux_json_data}
            else:
                yield {"event": "uiux_json_error", "message": "Structured UX plan extraction failed"}
        except Exception as _uje:
            yield {"event": "uiux_json_error", "message": str(_uje)}
    # ───────────────────────────────────────────────────────────────────────


# =============================================================================
# Frontend-only track: structured UXPlan extraction
# =============================================================================

_UIUX_JSON_SYSTEM = """\
You are a UX architect. Extract a structured UXPlan JSON from the UX design document below.
Return ONLY a valid JSON object — no markdown, no explanation — matching this schema:

{
  "screens": [
    {
      "screen_id": "S001",
      "layout": "dashboard|table|form|kanban|calendar|detail|split",
      "sections": ["section description 1", "section description 2"],
      "components_from_kit": ["AppShell", "StatCard", "DataTable"],
      "states": {
        "empty": "what user sees when no data",
        "loading": "spinner in content area",
        "error": "inline error with retry button"
      },
      "actions": ["primary action 1", "secondary action 2"],
      "navigation": "where main nav links go from this screen"
    }
  ]
}

Kit components available (use ONLY these names):
AppShell, PageHeader, Button, Card, StatCard, DataTable, FormField, Modal,
ConfirmDialog, Toast, EmptyState, LoadingState, ErrorState, Tabs,
LineChart, BarChart, PieChart, StatusChip

Rules:
- screen_id must match S001, S002, … from the PRD
- components_from_kit: list 2-6 components actually needed
- layout: pick the best single-word descriptor
- states: provide concrete copy for all three (empty/loading/error)"""


async def _emit_uiux_json(
    requirement: str,
    full_uiux: str,
    prd_json: Optional[Dict],
    llm: Any,
) -> Optional[Dict]:
    from app_builder.agents.structured_output import extract_validated
    from app_builder.schemas.frontend_plan import UXPlan

    screens_ctx = ""
    if prd_json and prd_json.get("screens"):
        screens_ctx = "Screens from PRD:\n" + "\n".join(
            f"- {s['id']}: {s['name']} — {s['purpose']}"
            for s in prd_json["screens"]
        ) + "\n\n"

    user_prompt = (
        f"{screens_ctx}"
        f"UX Design Document:\n\n{full_uiux}"
    )

    model = await extract_validated(
        llm=llm,
        system_prompt=_UIUX_JSON_SYSTEM,
        user_prompt=user_prompt,
        schema=UXPlan,
        max_retries=2,
        context_label="uiux_json",
    )
    if model is not None:
        return model.dict()
    return None
