from typing import List, Dict, Any, AsyncGenerator
import json
from langchain_core.messages import SystemMessage, HumanMessage

async def stream_uiux_generation(requirement: str, prd: str, llm) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streams UI/UX design generation.
    """
    system_prompt = """You are a World-Class UI/UX Designer for Enterprise SaaS Applications.
Your task is to design a PREMIUM, MODERN, and HIGHLY POLISHED interface based on the PRD.
The goal is to wow the user with a "Silicon Valley" standard design (think Stripe, Linear, Vercel).

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
- Primary: [Hex Code] - [Tailwind Name, e.g., indigo-600]
- Secondary: [Hex Code] - [Tailwind Name]
- Background: [Hex Code]
- Text: [Hex Code] - [Tailwind Name, e.g., slate-900]
- Muted/Gray: [Hex Code] - [Tailwind Name, e.g., slate-500]

## Typography
- Headings: [Font Family, e.g., Inter, Plus Jakarta Sans]
- Body: [Font Family]

## Layout Structure
- [Detailed description of the layout shell, e.g., "Collapsible Sidebar on left (w-64), Fixed Top Navbar (h-16), Scrollable Main Content area with max-w-7xl"]

## Key Screens
### 1. [Screen Name]
- **Layout**: [Grid/Flex structure]
- **Components**: 
  - [Component Name]: [Visual style, e.g., "Card with icon header and metric value"]
- **Interactions**: [Hover states, transitions]

(Repeat for 3-4 key screens)

## Components Styling Rules
- **Buttons**: px-4 py-2 rounded-md font-medium transition-colors.
- **Cards**: bg-white border border-gray-200 rounded-xl shadow-sm.
- **Inputs**: border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500.

keep it implementable in React + Tailwind CSS.
"""

    user_prompt = f"""Create a UI/UX Design for:

PRD Summary:
{prd}

Requirement:
{requirement}
"""

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
