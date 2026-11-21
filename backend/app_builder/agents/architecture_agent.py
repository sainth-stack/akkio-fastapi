import json
from typing import AsyncGenerator, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from ..schemas.architecture import ArchitectureDecision
from ..schemas.plan import ProjectPlan


async def stream_architecture_generation(
    requirement: str,
    prd: str,
    plan: list,
    llm,
    uiux: str = "",
    design_tokens: Optional[Dict[str, Any]] = None,
    builder_kind: Optional[str] = None,
    track: Optional[str] = None,
    prd_json: Optional[Dict[str, Any]] = None,
    uiux_json: Optional[Dict[str, Any]] = None,
    style_json: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Dynamically generates architecture decisions using LLM based on requirement, PRD, and plan.
    Streams progress events.
    """
    
    yield {
        "event": "architecture_start",
        "message": "Analyzing requirements and designing architecture..."
    }
    
    # Format plan for context
    plan_summary = "\n".join([
        f"{i+1}. {step.get('title', step.get('phase', 'Step'))}: {step.get('description', '')}"
        for i, step in enumerate(plan[:10])  # First 10 steps for context
    ])
    
    architecture_prompt = f"""Based on the following requirement, PRD, and implementation plan, design the optimal technical architecture.

**Requirement:**
{requirement}

**PRD Summary:**
{prd[:1000]}...

**Implementation Plan (first steps):**
{plan_summary}

**UI/UX Design Specification:**
{uiux if uiux else "Standard modern UI/UX design."}

**Design System Tokens (if available):**
{json.dumps(design_tokens, indent=2) if design_tokens else "Use UI/UX color palette — platform injects plain CSS (app-container, card, btn, input classes)."}

You must analyze the requirements and output a JSON architecture decision in this EXACT format:

```json
{{
  "frontend_structure": {{
    "framework": "React",
    "template": "Vite + React",
    "state_management": "Plain React (useState, useEffect) with LocalStorage fallback",
    "ui_library": "Plain CSS (platform-injected styles/app.css — classes: app-container, card, btn, input)",
    "key_components": ["<list main components>"],
    "static_data": ["frontend/src/data/*.json — seed JSON for demo/static content when PRD needs sample data"]
  }},
  "backend_structure": {{
    "framework": "FastAPI",
    "database": "SQLite",
    "orm": "SQLAlchemy",
    "api_style": "REST",
    "auth": "JWT",
    "key_services": ["<list main services>"]
  }},
  "database_schema": {{
    "tables": [
      {{
        "name": "<table name>",
        "columns": [
          {{"name": "id", "type": "string"}},
          {{"name": "<field>", "type": "<type>"}},
          {{"name": "created_at", "type": "datetime"}},
          {{"name": "updated_at", "type": "datetime"}}
        ]
      }}
    ]
  }},
  "screens": [
    {{"name": "<ScreenName>", "description": "<what this screen shows and what user can do>"}},
    {{"name": "<ScreenName>", "description": "<what this screen shows and what user can do>"}}
  ],
  "deployment": {{
    "containerization": "Docker"
  }},
  "rationale": "<brief explanation>",
  "project_structure": {{
    "backend": ["main.py", "requirements.txt", "database.py", "models.py", "schemas.py"],
    "frontend": ["package.json", "public/index.html", "src/index.js", "src/App.js", "src/styles.css"]
  }}
}}
```
CRITICAL RULES:
1. **Database MUST be SQLite**. Use SQLAlchemy with integer id primary keys.
2. **Use SQLAlchemy + SQLite**. No external database. Runs out of the box.
3. **Use Tailwind CSS for styling**. 
4. **Use Plain React State + LocalStorage**. NEVER use Zustand, Redux, or any external state manager.
5. Output ONLY the JSON, nothing else.
6. The 'screens' array MUST contain pages that directly correspond to what the user asked for.
   For example, if user asked for 'anomaly detection', screens should be: Dashboard, AnomalyFeed, AlertConfig, Reports.
   If user asked for 'inventory management', screens should be: Dashboard, Inventory, Orders, Suppliers, Reports.
   Do NOT include a Login screen — apps should be publicly accessible.
   Screen names must be descriptive and app-specific (not generic 'Page 1', 'Page 2').
   Include a 'screens' array in your JSON output with objects: {{"name": "<ScreenName>", "description": "<purpose>"}}.
"""

    from app_builder.services.fullstack_stack import is_fullstack, architecture_system_addendum, lock_architecture
    if is_fullstack(builder_kind):
        architecture_prompt = architecture_prompt + "\n" + architecture_system_addendum()

    messages = [
        SystemMessage(content="You are a senior solutions architect. Design optimal architectures based on requirements."),
        HumanMessage(content=architecture_prompt)
    ]
    
    buffer = ""
    
    yield {
        "event": "architecture_progress",
        "message": "Designing system architecture..."
    }
    
    async for chunk in llm.astream(messages):
        if hasattr(chunk, 'content'):
            buffer += chunk.content
    
    # Extract JSON from buffer
    architecture_data = None
    try:
        # Try to find JSON in code blocks
        if "```json" in buffer:
            json_start = buffer.find("```json") + 7
            json_end = buffer.find("```", json_start)
            json_str = buffer[json_start:json_end].strip()
            architecture_data = json.loads(json_str)
        elif "```" in buffer:
            json_start = buffer.find("```") + 3
            json_end = buffer.find("```", json_start)
            json_str = buffer[json_start:json_end].strip()
            architecture_data = json.loads(json_str)
        else:
            # Try to parse the whole buffer
            # Find first { and last }
            first_brace = buffer.find("{")
            last_brace = buffer.rfind("}")
            if first_brace != -1 and last_brace != -1:
                json_str = buffer[first_brace:last_brace+1]
                architecture_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        yield {
            "event": "architecture_progress",
            "message": f"JSON parsing failed, using intelligent fallback based on requirement..."
        }
        # Fallback: Create architecture based on requirement keywords
        architecture_data = create_fallback_architecture(requirement, prd, plan, builder_kind=builder_kind)
    
    if not architecture_data:
        architecture_data = create_fallback_architecture(requirement, prd, plan, builder_kind=builder_kind)

    if is_fullstack(builder_kind):
        architecture_data = lock_architecture(architecture_data)

    yield {
        "event": "architecture_complete",
        "data": architecture_data
    }

    # ── Frontend-only track: extract Blueprint JSON + cross-validate ───────
    if track == "frontend_only":
        try:
            yield {"event": "blueprint_json_start", "message": "Generating structured Blueprint…"}
            blueprint_json_data, cv_errors = await _emit_blueprint_json(
                requirement=requirement,
                prd=prd,
                uiux=uiux,
                architecture_data=architecture_data,
                prd_json=prd_json,
                uiux_json=uiux_json,
                style_json=style_json,
                llm=llm,
            )
            if blueprint_json_data:
                if cv_errors:
                    yield {"event": "blueprint_json_warning", "data": cv_errors}
                yield {"event": "blueprint_json", "data": blueprint_json_data}
            else:
                yield {"event": "blueprint_json_error", "message": "Blueprint extraction failed"}
        except Exception as _bje:
            yield {"event": "blueprint_json_error", "message": str(_bje)}
    # ───────────────────────────────────────────────────────────────────────


def create_fallback_architecture(
    requirement: str,
    prd: str,
    plan: list,
    builder_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates a reasonable architecture based on requirement keywords.
    """
    requirement_lower = requirement.lower()
    prd_lower = prd.lower()
    combined = requirement_lower + " " + prd_lower
    
    database = "SQLite"
    orm = "SQLAlchemy"
    
    # Determine frontend framework
    if "vue" in combined:
        framework = "Vue.js"
        state_management = "Pinia"
    elif "angular" in combined:
        framework = "Angular"
        state_management = "NgRx"
    else:
        framework = "React"
        state_management = "Plain React (useState) with LocalStorage"
    
    # Infer domain entities
    entities = infer_entities_from_requirement(requirement, prd)
    
    arch = {
        "frontend_structure": {
            "framework": framework,
            "template": "Modern SPA with component-based architecture",
            "state_management": state_management,
            "ui_library": "Tailwind CSS",
            "key_components": [comp for entity in entities for comp in [f"{entity.title()}List", f"{entity.title()}Detail", f"{entity.title()}Form"]] if entities else ["Dashboard", "ListView", "DetailView"]
        },
        "backend_structure": {
            "framework": "FastAPI",
            "database": database,
            "orm": orm,
            "api_style": "REST",
            "auth": "JWT with refresh tokens",
            "key_services": [f"{entity}_service" for entity in entities] if entities else ["data_service", "auth_service"]
        },
        "database_schema": {
            "tables": [
                {
                    "name": entity + "s",
                    "columns": [
                        {"name": "id", "type": "Integer", "primary_key": True},
                        {"name": "created_at", "type": "DateTime"},
                        {"name": "updated_at", "type": "DateTime"},
                        {"name": "name", "type": "String"},
                        {"name": "description", "type": "Text"}
                    ]
                }
                for entity in (entities[:3] if entities else ["item"])
            ]
        },
        "deployment": {
            "containerization": "Docker",
            "orchestration": "Docker Compose for development",
            "cloud_provider": "Cloud-agnostic"
        },
        "rationale": f"Architecture designed for {requirement}. Using {framework} for modern UI, FastAPI for high-performance backend, and {database} for data persistence.",
        "project_structure": {
            "backend": ["main.py", "requirements.txt", "database.py", "models.py", "schemas.py"],
            "frontend": ["package.json", "public/index.html", "src/index.js", "src/App.js", "src/styles.css"]
        }
    }
    from app_builder.services.fullstack_stack import is_fullstack, lock_architecture
    if is_fullstack(builder_kind):
        return lock_architecture(arch)
    return arch


def infer_entities_from_requirement(requirement: str, prd: str) -> list:
    """
    Attempts to infer main entities from requirement text.
    """
    combined = (requirement + " " + prd).lower()
    
    # Common domain patterns
    entity_keywords = {
        "user": ["user", "account", "profile", "member"],
        "product": ["product", "item", "goods", "merchandise"],
        "order": ["order", "purchase", "transaction", "sale"],
        "post": ["post", "article", "blog", "content"],
        "comment": ["comment", "reply", "feedback"],
        "task": ["task", "todo", "assignment", "job"],
        "project": ["project", "workspace"],
        "ticket": ["ticket", "issue", "support"],
        "event": ["event", "appointment", "booking"],
        "message": ["message", "chat", "conversation"],
        "book": ["book", "library", "publication", "volume"]
    }
    
    found_entities = []
    for entity, keywords in entity_keywords.items():
        if any(keyword in combined for keyword in keywords):
            found_entities.append(entity)
    
    return found_entities[:3] if found_entities else ["item"]


def architecture_agent(plan: ProjectPlan, requirement: str = "") -> ArchitectureDecision:
    """
    Legacy sync function - updated to be more dynamic.
    Returns an architecture decision with entities inferred from the requirement/plan.
    """
    plan_text = " ".join(plan.steps)
    combined = (requirement + " " + plan_text).lower()
    
    entities = infer_entities_from_requirement(requirement, plan_text)
    
    return ArchitectureDecision(
        frontend_structure={
            "framework": "React",
            "template": "Create React App",
            "state_management": "Plain React (useState) with LocalStorage",
            "ui_library": "Tailwind CSS"
        },
        backend_structure={
            "framework": "FastAPI",
            "database": "SQLite",
            "orm": "SQLAlchemy"
        },
        database_schema={"tables": []},
        deployment={},
        rationale=f"Dynamic architecture for {requirement or 'app'}.",
        project_structure={
            "backend": ["main.py", "requirements.txt", "database.py", "models.py", "schemas.py"],
            "frontend": ["package.json", "public/index.html", "src/index.js", "src/App.js", "src/styles.css"]
        }
    )


# =============================================================================
# Frontend-only track: structured Blueprint extraction + cross-validation
# =============================================================================

_BLUEPRINT_JSON_SYSTEM = """\
You are a frontend architect. Produce a Blueprint JSON for a React SPA.
Return ONLY valid JSON — no markdown, no explanation — matching this schema exactly:

{
  "routes": [
    {"path": "/", "page": "DashboardPage", "screen_id": "S001"}
  ],
  "pages": [
    {
      "file": "DashboardPage",
      "imports": ["AppShell", "StatCard", "DataTable"],
      "uses_entities": ["Deal", "Activity"],
      "uses_kit_components": ["AppShell", "StatCard", "DataTable"]
    }
  ],
  "entities": [
    {
      "name": "Deal",
      "fields": [
        {"name": "id", "type": "string"},
        {"name": "title", "type": "string"},
        {"name": "amount", "type": "number"},
        {"name": "stage", "type": "string"},
        {"name": "createdAt", "type": "date"}
      ]
    }
  ],
  "mock_plan": [
    {"entity": "Deal", "row_count": 20, "value_hints": ["stage: Prospecting|Proposal|Closed Won|Closed Lost"]}
  ],
  "nav_items": [
    {"label": "Dashboard", "path": "/", "icon": "Dashboard"}
  ],
  "file_order": ["types", "mock", "theme", "layout", "shared", "pages", "app"]
}

Kit components available (use ONLY these names):
AppShell, PageHeader, Button, Card, StatCard, DataTable, FormField, Modal,
ConfirmDialog, Toast, EmptyState, LoadingState, ErrorState, Tabs,
LineChart, BarChart, PieChart, StatusChip

Rules:
- routes: one entry per screen; path must be unique; / is the default
- pages: file is the component name without .tsx; imports ⊆ kit names
- entities: PascalCase singular; include all entities used by any page
- mock_plan: one entry per entity; row_count 10-50; value_hints = domain-realistic
- nav_items: icon = any MUI icon name (PascalCase, e.g. Dashboard, People, BarChart)
- file_order: always exactly ["types","mock","theme","layout","shared","pages","app"]"""


async def _emit_blueprint_json(
    requirement: str,
    prd: str,
    uiux: str,
    architecture_data: Any,
    prd_json: Optional[Dict],
    uiux_json: Optional[Dict],
    style_json: Optional[Dict],
    llm: Any,
) -> tuple:
    """
    Returns (blueprint_dict_or_None, list_of_cross_validation_errors).
    Applies cross-validation and one auto-repair attempt if needed.
    """
    from app_builder.agents.structured_output import extract_validated, call_llm_once
    from app_builder.schemas.frontend_plan import Blueprint, cross_validate_blueprint, repair_blueprint_prompt

    # Build context from prior JSON contracts
    ctx_parts = [f"Requirement: {requirement}"]
    if prd_json:
        ctx_parts.append("PRD screens:\n" + "\n".join(
            f"- {s['id']}: {s['name']} (entities: {s.get('entities', [])})"
            for s in prd_json.get("screens", [])
        ))
    if uiux_json:
        ctx_parts.append("UX per screen:\n" + "\n".join(
            f"- {sx['screen_id']}: layout={sx['layout']}, kit={sx.get('components_from_kit', [])}"
            for sx in uiux_json.get("screens", [])
        ))

    user_prompt = "\n\n".join(ctx_parts)

    model = await extract_validated(
        llm=llm,
        system_prompt=_BLUEPRINT_JSON_SYSTEM,
        user_prompt=user_prompt,
        schema=Blueprint,
        max_retries=2,
        context_label="blueprint_json",
    )
    if model is None:
        return None, []

    # Cross-validate
    prd_model = None
    if prd_json:
        try:
            from app_builder.schemas.frontend_plan import PRDPlan
            prd_model = PRDPlan(**prd_json)
        except Exception:
            pass

    cv_errors = cross_validate_blueprint(model, prd=prd_model)

    if cv_errors:
        import logging
        logging.getLogger("app_builder").warning(
            "[blueprint_json] cross-validation failed (%d errors); attempting auto-repair", len(cv_errors)
        )
        repair_prompt = repair_blueprint_prompt(model.dict(), cv_errors)
        repair_system = _BLUEPRINT_JSON_SYSTEM + "\n\nFix the blueprint to satisfy the listed errors."
        try:
            from app_builder.agents.structured_output import extract_validated
            repaired = await extract_validated(
                llm=llm,
                system_prompt=repair_system,
                user_prompt=repair_prompt,
                schema=Blueprint,
                max_retries=1,
                context_label="blueprint_json_repair",
            )
            if repaired:
                remaining = cross_validate_blueprint(repaired, prd=prd_model)
                if len(remaining) < len(cv_errors):
                    model = repaired
                    cv_errors = remaining
        except Exception:
            pass  # keep original if repair fails

    return model.dict(), cv_errors
