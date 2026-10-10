"""
PRD Agent - Generates Product Requirements Document and detailed plan using LLM.

For the frontend-only track this agent also emits a validated PRDPlan JSON contract
(event: "prd_json") after the markdown stream finishes.
"""
from typing import Dict, Any, AsyncGenerator, Optional
import json
from langchain_core.messages import HumanMessage, SystemMessage


def _chunk_text(content) -> str:
    """Normalize LangChain chunk content to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return "".join(parts)
    return str(content)


async def stream_prd_generation(
    requirement: str,
    llm,
    skip_plan: bool = False,
    builder_kind: Optional[str] = None,
    track: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streams PRD generation using LLM.
    
    Args:
        requirement: User requirement description
        llm: Initialized LLM instance
        skip_plan: If True, stops after PRD generation
    """
    
    system_prompt = """You are a Product Requirements Document (PRD) specialist.
Your task is to create a CONCISE and EFFECTIVE PRD with perfect structure and formatting.
Use clean Markdown so the document displays correctly. Avoid unnecessary details.

IMPORTANT: The 'Key Features' and 'User Stories' sections MUST directly reflect what the user described in their prompt.
Do not add features the user didn't ask for. Do not include authentication/login unless the user explicitly mentioned it.

You MUST follow this exact structure. Use these exact headings and spacing:

# 1. Product Overview

[One clear sentence about the app.]

# 2. Business Requirements

## 2.1 Functional Requirements (FR)

- [Bullet list of functional requirements: what the system must do]
- [One requirement per bullet; be specific and actionable]

## 2.2 Non-Functional Requirements (NFR)

- [Bullet list of non-functional requirements: performance, security, usability, scalability, etc.]
- [One requirement per bullet]

# 3. Core Features

- [Bulleted list of essential features]
- [One feature per bullet]

# 4. Process Flows / Use Cases

[For each main user flow or use case, describe briefly:]
- **Use Case 1:** [Actor] – [Goal/Action] – [Outcome]
- **Use Case 2:** [Actor] – [Goal/Action] – [Outcome]
[Add 3–6 key use cases as needed]

Formatting rules:
- Use exactly one blank line between sections.
- Use `#` for main sections (1–4), `##` for subsections (2.1, 2.2).
- Use `- ` for all bullet lists; no mixed styles.
- Keep sentences short and clear. No run-on paragraphs.
- Do not add a Tech Stack section unless the product is a fullstack enterprise application."""

    user_prompt = f"""Create a concise and well-formatted PRD for:

{requirement}

Follow the exact structure (1. Product Overview, 2. Business Requirements with FR and NFR, 3. Core Features, 4. Process Flows / Use Cases). Use clean Markdown so it displays perfectly."""

    from app_builder.services.fullstack_stack import is_fullstack, prd_system_addendum
    if is_fullstack(builder_kind):
        system_prompt = system_prompt + "\n" + prd_system_addendum()
        user_prompt += (
            "\n\nThis is a FULLSTACK agentic application. Include modules, APIs, data model, "
            "roles, quality requirements, and deliverables. Keep the locked tech stack."
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # Stream the PRD generation
    full_prd = ""
    async for chunk in llm.astream(messages):
        if hasattr(chunk, 'content'):
            content = _chunk_text(chunk.content)
            if not content:
                continue
            full_prd += content
            yield {
                "event": "prd_chunk",
                "data": content
            }
    
    if not full_prd.strip():
        yield {
            "event": "error",
            "message": "PRD generation returned empty content. Check your LLM API key and model settings.",
        }
        return

    # Parse the plan from the PRD
    yield {
        "event": "prd_complete",
        "data": full_prd
    }

    # ── Frontend-only track: extract structured PRDPlan JSON ────────────────
    if track == "frontend_only":
        try:
            yield {"event": "prd_json_start", "message": "Extracting structured plan…"}
            prd_json_data = await _emit_prd_json(requirement, full_prd, llm)
            if prd_json_data:
                yield {"event": "prd_json", "data": prd_json_data}
            else:
                yield {"event": "prd_json_error", "message": "Structured plan extraction failed (will use markdown only)"}
        except Exception as _pje:
            yield {"event": "prd_json_error", "message": str(_pje)}
    # ───────────────────────────────────────────────────────────────────────

    if skip_plan:
        yield { "event": "done", "message": "PRD generation complete" }
        return

    # Now generate structured plan with streaming
    yield {
        "event": "plan_start",
        "message": "Generating structured plan..."
    }
    
    plan_prompt = f"""Based on the following PRD, create a detailed implementation plan with specific, actionable steps.

{full_prd}

You must provide 5-8 detailed steps. For EACH step, output it immediately in this exact format:

STEP: {{"id": <number>, "title": "<brief title>", "description": "<detailed description>", "status": "pending", "dependencies": [<array of step numbers this depends on>]}}

Requirements:
- Output each step immediately as you think of it
- Start with foundational setup steps
- Include dependencies on previous steps where logical
- Cover: setup, data models, backend, frontend, integration, testing
- Be specific and actionable

Start now with STEP 1:"""

    plan_messages = [
        SystemMessage(content="You are a technical project planner. Output each step immediately as you generate it."),
        HumanMessage(content=plan_prompt)
    ]
    
    plan_buffer = ""
    collected_steps = []
    
    async for chunk in llm.astream(plan_messages):
        if hasattr(chunk, 'content'):
            plan_buffer += _chunk_text(chunk.content)
            
            # Look for complete STEP: {...} patterns
            while "STEP:" in plan_buffer:
                step_start = plan_buffer.find("STEP:")
                # Find the JSON object
                json_start = plan_buffer.find("{", step_start)
                if json_start == -1:
                    break
                
                # Find matching closing brace
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(plan_buffer)):
                    if plan_buffer[i] == '{':
                        brace_count += 1
                    elif plan_buffer[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end == -1:
                    break
                
                try:
                    step_json = plan_buffer[json_start:json_end]
                    step = json.loads(step_json)
                    collected_steps.append(step)
                    
                    # Stream this step immediately
                    yield {
                        "event": "plan_step",
                        "data": step
                    }
                    
                    plan_buffer = plan_buffer[json_end:]
                except json.JSONDecodeError:
                    # Move past this STEP marker and try again
                    plan_buffer = plan_buffer[step_start + 5:]
    
    # If we got steps, send them all as complete
    if collected_steps:
        yield {
            "event": "plan_complete",
            "data": collected_steps
        }
    else:
        # Fallback to basic plan
        fallback_plan = [
            {"id": 1, "title": "Initialize project structure", "description": "Set up development environment with required frameworks and tools", "status": "pending", "dependencies": []},
            {"id": 2, "title": "Design database schema", "description": "Create entity models and database structure", "status": "pending", "dependencies": [1]},
            {"id": 3, "title": "Implement data layer", "description": "Set up ORM and database connection", "status": "pending", "dependencies": [2]},
            {"id": 4, "title": "Build backend API endpoints", "description": "Create REST API for all operations", "status": "pending", "dependencies": [3]},
            {"id": 5, "title": "Setup frontend framework", "description": "Initialize React application structure", "status": "pending", "dependencies": [1]},
            {"id": 6, "title": "Create UI components", "description": "Build reusable UI components", "status": "pending", "dependencies": [5]},
            {"id": 7, "title": "Implement main views", "description": "Create application screens and layouts", "status": "pending", "dependencies": [6]},
            {"id": 8, "title": "Integrate frontend with backend", "description": "Connect UI to API endpoints", "status": "pending", "dependencies": [4, 7]},
            {"id": 9, "title": "Add authentication", "description": "Implement user authentication system", "status": "pending", "dependencies": [8]},
            {"id": 10, "title": "Implement core features", "description": "Build main application functionality", "status": "pending", "dependencies": [9]},
            {"id": 11, "title": "Add error handling", "description": "Implement comprehensive error handling", "status": "pending", "dependencies": [10]},
            {"id": 12, "title": "Testing and debugging", "description": "Test all features and fix issues", "status": "pending", "dependencies": [11]}
        ]
        
        # Stream fallback steps one by one
        for step in fallback_plan:
            yield {
                "event": "plan_step",
                "data": step
            }
        
        yield {
            "event": "plan_complete",
            "data": fallback_plan
        }
    
    yield {
        "event": "done",
        "message": "PRD and plan generation complete"
    }


# =============================================================================
# Frontend-only track: structured PRDPlan extraction
# =============================================================================

# ── Few-shot examples (domain-aware) ─────────────────────────────────────────

_FEW_SHOT_EXAMPLES = {
    "calculator": """\
Example — Calculator (simple tool, 1 screen):
{
  "summary": "A browser-based scientific calculator for arithmetic and basic math functions.",
  "personas": ["Student", "Developer"],
  "goals": ["Perform arithmetic instantly", "Support keyboard input"],
  "features": [
    {"id": "F001", "name": "Basic arithmetic (+−×÷)", "priority": "must"},
    {"id": "F002", "name": "Clear and backspace", "priority": "must"},
    {"id": "F003", "name": "Calculation history", "priority": "should"}
  ],
  "screens": [
    {"id": "S001", "name": "Calculator", "purpose": "Enter and evaluate arithmetic expressions", "entities": []}
  ],
  "non_goals": ["User accounts", "Sync across devices", "Advanced CAS features"]
}""",

    "crm": """\
Example — CRM (5 screens, real domain entities):
{
  "summary": "A CRM for managing sales contacts, deals, and pipeline stages.",
  "personas": ["Sales rep", "Sales manager"],
  "goals": ["Track leads through pipeline", "Log activities on contacts", "Forecast revenue"],
  "features": [
    {"id": "F001", "name": "Contact management (create/edit/search)", "priority": "must"},
    {"id": "F002", "name": "Deal tracking with pipeline stages", "priority": "must"},
    {"id": "F003", "name": "Activity log (calls, emails, meetings)", "priority": "must"},
    {"id": "F004", "name": "Revenue dashboard with KPIs", "priority": "must"},
    {"id": "F005", "name": "CSV import for bulk contacts", "priority": "should"}
  ],
  "screens": [
    {"id": "S001", "name": "Dashboard", "purpose": "KPIs: open deals, closed-won, activities due today", "entities": ["Deal", "Activity"]},
    {"id": "S002", "name": "Contacts", "purpose": "Searchable table of contacts with inline edit", "entities": ["Contact"]},
    {"id": "S003", "name": "Deals", "purpose": "Kanban board of deals grouped by pipeline stage", "entities": ["Deal", "PipelineStage"]},
    {"id": "S004", "name": "Activities", "purpose": "Calendar + list of upcoming and past activities", "entities": ["Activity", "Contact"]},
    {"id": "S005", "name": "Reports", "purpose": "Revenue by month bar chart, win/loss pie chart", "entities": ["Deal"]}
  ],
  "non_goals": ["Email sending", "Calendar sync", "Mobile app"]
}""",

    "analytics_dashboard": """\
Example — Analytics Dashboard (4 screens):
{
  "summary": "A real-time business intelligence dashboard visualising KPIs and trends.",
  "personas": ["Business analyst", "C-suite executive"],
  "goals": ["Surface key metrics at a glance", "Drill into trends by date range", "Export reports as CSV"],
  "features": [
    {"id": "F001", "name": "KPI stat cards (revenue, users, orders)", "priority": "must"},
    {"id": "F002", "name": "Line chart — metric over time with date picker", "priority": "must"},
    {"id": "F003", "name": "Bar chart — comparison by category", "priority": "must"},
    {"id": "F004", "name": "Data table with sorting and CSV export", "priority": "should"}
  ],
  "screens": [
    {"id": "S001", "name": "Overview", "purpose": "4 KPI cards + sparkline trends", "entities": ["Metric"]},
    {"id": "S002", "name": "Trends", "purpose": "Full-page line chart with date-range selector", "entities": ["MetricSeries"]},
    {"id": "S003", "name": "Breakdown", "purpose": "Bar/pie charts segmented by category", "entities": ["MetricSegment"]},
    {"id": "S004", "name": "Raw Data", "purpose": "Paginated sortable table of underlying data", "entities": ["MetricRecord"]}
  ],
  "non_goals": ["Live streaming data", "Custom SQL queries", "Multi-tenant workspaces"]
}""",

    "kanban": """\
Example — Kanban board (3 screens):
{
  "summary": "A Trello-style kanban for managing tasks across customisable columns.",
  "personas": ["Product manager", "Engineer"],
  "goals": ["Drag cards between columns", "Track task status visually", "Assign and label cards"],
  "features": [
    {"id": "F001", "name": "Multiple boards", "priority": "must"},
    {"id": "F002", "name": "Drag-and-drop card movement", "priority": "must"},
    {"id": "F003", "name": "Card detail: description, due date, assignee, labels", "priority": "must"},
    {"id": "F004", "name": "Board search and filter by label/assignee", "priority": "should"}
  ],
  "screens": [
    {"id": "S001", "name": "Boards", "purpose": "Grid of all boards with create/archive", "entities": ["Board"]},
    {"id": "S002", "name": "Board View", "purpose": "Columns of cards with drag-and-drop", "entities": ["Column", "Card"]},
    {"id": "S003", "name": "Card Detail", "purpose": "Full-page card with comments, attachments, history", "entities": ["Card", "Comment"]}
  ],
  "non_goals": ["Real-time collaboration", "OAuth login", "Calendar view"]
}""",

    "booking": """\
Example — Booking system (4 screens):
{
  "summary": "An appointment booking system for service businesses.",
  "personas": ["Customer booking a service", "Business owner managing appointments"],
  "goals": ["Show real-time availability", "Allow online booking in <3 steps", "Send booking confirmation"],
  "features": [
    {"id": "F001", "name": "Service catalogue with duration and price", "priority": "must"},
    {"id": "F002", "name": "Calendar availability picker", "priority": "must"},
    {"id": "F003", "name": "Booking form (name, email, notes)", "priority": "must"},
    {"id": "F004", "name": "My Bookings — view/cancel upcoming appointments", "priority": "should"}
  ],
  "screens": [
    {"id": "S001", "name": "Services", "purpose": "Cards of available services with select button", "entities": ["Service"]},
    {"id": "S002", "name": "Availability", "purpose": "Month calendar + time slot selector", "entities": ["TimeSlot"]},
    {"id": "S003", "name": "Booking Form", "purpose": "Contact details + confirm booking", "entities": ["Booking"]},
    {"id": "S004", "name": "My Bookings", "purpose": "Table of upcoming and past bookings with cancel", "entities": ["Booking", "Service"]}
  ],
  "non_goals": ["Payment processing", "SMS reminders", "Multi-staff scheduling"]
}""",

    "inventory": """\
Example — Inventory management (5 screens):
{
  "summary": "A warehouse inventory management app tracking SKUs, stock levels and supplier orders.",
  "personas": ["Warehouse manager", "Procurement officer"],
  "goals": ["Track stock levels in real time", "Raise and track purchase orders", "Alert on low stock"],
  "features": [
    {"id": "F001", "name": "Product / SKU catalogue with stock counts", "priority": "must"},
    {"id": "F002", "name": "Stock adjustments (receive, dispatch, write-off)", "priority": "must"},
    {"id": "F003", "name": "Supplier management", "priority": "must"},
    {"id": "F004", "name": "Purchase order creation and status tracking", "priority": "must"},
    {"id": "F005", "name": "Low-stock dashboard with reorder alerts", "priority": "should"}
  ],
  "screens": [
    {"id": "S001", "name": "Dashboard", "purpose": "Low-stock alerts + KPIs: total SKUs, POs in transit", "entities": ["Product", "PurchaseOrder"]},
    {"id": "S002", "name": "Products", "purpose": "Searchable table of SKUs with stock-level badges", "entities": ["Product"]},
    {"id": "S003", "name": "Suppliers", "purpose": "Supplier list with contact info and rating", "entities": ["Supplier"]},
    {"id": "S004", "name": "Purchase Orders", "purpose": "Create, track, and receive POs", "entities": ["PurchaseOrder", "Supplier", "Product"]},
    {"id": "S005", "name": "Reports", "purpose": "Stock movement history chart, top products", "entities": ["Product", "StockMovement"]}
  ],
  "non_goals": ["Barcode scanning hardware integration", "Multi-warehouse sync", "Invoicing"]
}""",
}

_DOMAIN_KEYWORDS: Dict[str, list] = {
    "calculator": ["calculator", "arithmetic", "math", "calc"],
    "crm": ["crm", "contacts", "leads", "deals", "sales pipeline", "customer relationship"],
    "analytics_dashboard": ["analytics", "dashboard", "kpi", "business intelligence", "bi ", "reporting"],
    "kanban": ["kanban", "trello", "board", "task management", "sprint", "columns"],
    "booking": ["booking", "appointment", "scheduling", "availability", "reservation"],
    "inventory": ["inventory", "warehouse", "stock", "sku", "supply chain", "procurement"],
}


def _pick_few_shot(requirement: str) -> str:
    """Return 1-2 relevant few-shot examples based on keyword matching."""
    req_lower = requirement.lower()
    picked = []
    for domain, kws in _DOMAIN_KEYWORDS.items():
        if any(kw in req_lower for kw in kws):
            picked.append(_FEW_SHOT_EXAMPLES[domain])
            if len(picked) == 2:
                break
    if not picked:
        # Default: CRM + inventory as generic examples
        picked = [_FEW_SHOT_EXAMPLES["crm"], _FEW_SHOT_EXAMPLES["inventory"]]
    return "\n\n".join(picked)


_PRD_JSON_SYSTEM = """\
You are a requirements analyst. Extract a structured PRDPlan JSON from the PRD below.
Return ONLY a valid JSON object matching this schema — no markdown, no explanation:

{
  "summary": "one-sentence summary of the app",
  "personas": ["primary user type 1", ...],
  "goals": ["goal 1", ...],
  "features": [
    {"id": "F001", "name": "feature name", "priority": "must|should|could"},
    ...
  ],
  "screens": [
    {"id": "S001", "name": "Screen Name", "purpose": "what user sees/does", "entities": ["EntityName", ...]},
    ...
  ],
  "non_goals": ["thing explicitly NOT in scope", ...]
}

Rules:
- Feature ids: F001, F002, … (sequential)
- Screen ids: S001, S002, … (sequential)
- Simple tools (calculator, timer): 1-2 screens
- CRUD apps (CRM, inventory): 4-6 screens — include Dashboard, list pages, reports
- entities in screens: use PascalCase singular nouns matching the domain (Lead, Deal, Product, …)
- non_goals: list 2-4 explicit exclusions to bound scope"""


async def _emit_prd_json(
    requirement: str,
    full_prd: str,
    llm: Any,
) -> Optional[Dict]:
    """
    Extract + validate a PRDPlan JSON from the generated PRD markdown.
    Returns the dict (ready to yield as plan_json event), or None on failure.
    """
    from app_builder.agents.structured_output import extract_validated
    from app_builder.schemas.frontend_plan import PRDPlan

    few_shot = _pick_few_shot(requirement)
    user_prompt = (
        f"Few-shot examples for reference:\n\n{few_shot}\n\n"
        f"---\nNow extract a PRDPlan JSON for the following PRD:\n\n{full_prd}"
    )

    model = await extract_validated(
        llm=llm,
        system_prompt=_PRD_JSON_SYSTEM,
        user_prompt=user_prompt,
        schema=PRDPlan,
        max_retries=2,
        context_label="prd_json",
    )
    if model is not None:
        return model.dict()
    return None


async def generate_prd_json(
    requirement: str,
    full_prd: str,
    llm: Any,
) -> Optional[Dict]:
    """Public entry point for PRDPlan extraction (usable by tests and the API)."""
    return await _emit_prd_json(requirement, full_prd, llm)
