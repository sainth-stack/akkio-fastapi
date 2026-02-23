import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from ..schemas.requirements import UserRequirement

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
  "project_name": "string",
  "frontend": "react",
  "backend": "fastapi",
  "database": "mongodb",
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
2. Use 'mongodb' as the database. NEVER use 'sqlite'.
3. Be exhaustive in listing fields based on the description.
4. Output ONLY the raw JSON string.
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
        return data
    except json.JSONDecodeError:
        # Minimal fallback
        return {
            "project_name": "app",
            "frontend": "react",
            "backend": "fastapi",
            "database": "mongodb",
            "entities": [],
            "features": []
        }
