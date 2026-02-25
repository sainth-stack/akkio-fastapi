import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

async def api_contract_agent(structured_requirement: Dict[str, Any], blueprint: Dict[str, Any], llm=None) -> Dict[str, Any]:
    """
    API Contract Agent - Defines the strict request/response format for all endpoints.
    Both frontend and backend MUST follow this.
    """
    if llm is None:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0)

    system_prompt = """You are an API Designer. Your task is to define a strict API contract based on structured requirements and a project blueprint.
Output ONLY a JSON contract. No creativity. No markdown.

Contract Structure:
{
  "endpoints": {
    "METHOD /path": {
      "request": {
        "field_name": "type"
      },
      "response": {
        "field_name": "type"
      },
      "errors": {
        "404": "Not Found",
        "400": "Bad Request"
      }
    }
  }
}

CRITICAL RULES:
1. Define every route listed in the blueprint.
2. Ensure types are explicit (string, integer, boolean, etc.).
3. Database is SQLite: 'id' fields are integer. Use integer for IDs in paths.
4. Request bodies for POST/PUT must be detailed.
5. Response bodies must include the full entity structure.
6. Output ONLY the JSON.
"""

    context = {
        "structured_requirement": structured_requirement,
        "blueprint": blueprint
    }
    
    user_prompt = f"Context: {json.dumps(context, indent=2)}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    content = response.content.strip()
    
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        return {
            "error": "Failed to generate API contract"
        }
