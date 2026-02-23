"""
Requirement Agent - Fetches all details from PRD and UI/UX design,
produces a minimal end-to-end consolidated summary for downstream generation.
"""
from ..schemas.requirements import UserRequirement
from langchain_core.messages import SystemMessage, HumanMessage
from ..services.template_service import detect_template, load_template



async def _build_consolidated_summary_async(requirement: str, prd: str, uiux: str, llm) -> str:
    """
    Uses LLM to consolidate requirement + PRD + UI/UX into a minimal end-to-end summary.
    """
    system_prompt = """You are a requirements analyst. Your task is to consolidate the user requirement, PRD, and UI/UX design into a SINGLE minimal, end-to-end specification.

Output format:
1. **Core Requirement** (1-2 sentences): What the app is and its main purpose.
2. **Key Features** (bullet list): Essential features from PRD—only the most important 4-8 items.
3. **UI/UX Summary** (2-4 bullets): Layout, colors, components, and interaction patterns from the UI/UX design.

Keep the entire output concise—under 250 words. Use the exact structure above with clear section headings. No fluff."""
    
    prd_snippet = (prd[:4000] + "...") if len(prd) > 4000 else prd
    uiux_snippet = (uiux[:2000] + "...") if len(uiux) > 2000 else (uiux if uiux else "Standard modern UI/UX design.")
    
    user_prompt = f"""Consolidate these into a minimal end-to-end specification:

**User Requirement:**
{requirement.strip()}

**PRD (Product Requirements):**
{prd_snippet}

**UI/UX Design:**
{uiux_snippet}"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    if hasattr(response, 'content'):
        return response.content.strip()
    return str(response).strip()


async def requirement_agent(
    requirement: UserRequirement,
    prd: str = "",
    uiux: str = "",
    llm=None
) -> str:
    """
    Fetches all details from PRD and UI/UX design, produces a minimal end-to-end
    consolidated summary that downstream agents use for generation.
    """
    desc = (requirement.description or "").strip()
    if not desc:
        return ""
    
    # Check for template matches early
    template_name = detect_template(desc)
    if template_name:
        template_data = load_template(template_name)
        if template_data:
            app_name = template_data.get("app_name", template_name)
            return f"TEMPLATE_DETECTED: {template_name} | APP_NAME: {app_name}"
    
    # When PRD or UI/UX exists, consolidate everything via LLM
    if (prd and prd.strip()) or (uiux and uiux.strip()):
        if llm is None:
            from llm_helper import get_llm_for_user
            llm = get_llm_for_user(user_email=None, temperature=0.2)
        return await _build_consolidated_summary_async(desc, prd or "", uiux or "", llm)
    
    # Fallback: no PRD/UI/UX—just normalize the requirement
    return " ".join(desc.split())

