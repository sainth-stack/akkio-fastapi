import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

async def database_schema_agent(structured_requirement: Dict[str, Any], llm=None) -> str:
    """
    Database Schema Agent - Converts entity JSON into SQLite table definitions.
    """
    if llm is None:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0)

    system_prompt = """You are a Database Architect. Define SQLite tables based on entity definitions.
Output a concise description of tables, columns, types (Integer, String, Boolean, DateTime, Text, Float), and primary/foreign keys.

CRITICAL RULES:
1. Use integer primary keys. Include created_at, updated_at (DateTime) where needed.
2. Output ONLY the description. No markdown block.
"""

    user_prompt = f"Entities: {json.dumps(structured_requirement.get('entities', []), indent=2)}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    content = response.content.strip()
    
    # Clean up markdown if LLM misbehaves
    if content.startswith("```sql"):
        content = content[6:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    return content
