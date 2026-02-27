import json
import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from ..schemas.requirements import UserRequirement
from ..services.template_service import detect_template

logger = logging.getLogger("app_builder")

async def requirement_structuring_agent(requirement: UserRequirement, llm=None) -> Dict[str, Any]:
    """
    Requirement Structuring Agent - Converts raw requirement text into a strict structured JSON.
    This becomes the single source of truth for the entire pipeline.
    """
    if llm is None:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0) # Strict output

    system_prompt = """You are a software requirements architect. Your task is to convert unstructured user requirements into a STRICT JSON format.
No creativity. No markdown. Only the following structure:

{
  "project_name": "string (use hyphens, no spaces, e.g. todo-list-application)",
  "template_name": "string or null (ideas-generator for LLM/GenAI apps, todo-list for normal CRUD apps - set null if no match)",
  "frontend": "react",
  "backend": "fastapi",
  "database": "sqlite",
  "entities": [
    {
      "name": "string (Singular PascalCase, e.g., Todo)",
      "table_name": "string (plural underscore_case, e.g., todos)",
      "fields": [
        {"name": "string", "type": "integer/string/boolean/datetime/float", "primary": boolean, "required": boolean}
      ]
    }
  ],
  "features": [
    "string (short description of feature, e.g., create_todo)"
  ]
}

CRITICAL RULES:
1. Always include an 'id' field as primary key for every entity.
2. Use 'sqlite' as the database. No external DB required.
3. Be exhaustive in listing fields based on the description.
4. Output ONLY the raw JSON string.
5. project_name MUST use hyphens not spaces (e.g. todo-list-application, travel-planner) to avoid path issues.
6. template_name: ideas-generator for LLM apps (ideas, linkedin post, travel planner, translator, genai). todo-list for todo/task/checklist. Otherwise null.
"""

    user_prompt = f"Requirement: {requirement.description}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    content = response.content.strip()
    
    # Clean up markdown if LLM misbehaves
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    try:
        data = json.loads(content)
        desc = requirement.description if hasattr(requirement, "description") else str(requirement)
        template_name = detect_template(desc)
        if not template_name:
            template_name = detect_template(data.get("project_name", "") or "")
        data["template_name"] = template_name
        logger.info("[structuring_agent] parsed | project_name=%s | template_name=%s", data.get("project_name"), template_name)
        return data
    except json.JSONDecodeError:
        # Minimal fallback
        template_name = detect_template(requirement.description if hasattr(requirement, "description") else str(requirement))
        return {
            "project_name": "app",
            "template_name": template_name,
            "frontend": "react",
            "backend": "fastapi",
            "database": "sqlite",
            "entities": [],
            "features": []
        }
