import json
from typing import AsyncGenerator, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from ..schemas.architecture import ArchitectureDecision
from ..schemas.plan import ProjectPlan


async def stream_architecture_generation(
    requirement: str,
    prd: str,
    plan: list,
    llm,
    uiux: str = ""
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

You must analyze the requirements and output a JSON architecture decision in this EXACT format:

```json
{{
  "frontend_structure": {{
    "framework": "React",
    "template": "Create React App (No Zustand)",
    "state_management": "Plain React (useState, useEffect) with LocalStorage",
    "ui_library": "Tailwind CSS",
    "key_components": ["<list main components>"]
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
"""

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
        architecture_data = create_fallback_architecture(requirement, prd, plan)
    
    if not architecture_data:
        architecture_data = create_fallback_architecture(requirement, prd, plan)
    
    yield {
        "event": "architecture_complete",
        "data": architecture_data
    }


def create_fallback_architecture(requirement: str, prd: str, plan: list) -> Dict[str, Any]:
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
    
    return {
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
