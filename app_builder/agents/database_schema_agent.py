import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

async def database_schema_agent(structured_requirement: Dict[str, Any], llm=None) -> str:
    """
    Database Schema Agent - Converts entity JSON into MongoDB collection definitions.
    No backend logic.
    """
    if llm is None:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0)

    system_prompt = """You are a Database Architect. Your task is to define the structure for MongoDB collections based on entity definitions.
Output a concise description of the collections, their fields, and any indexes that should be created.

CRITICAL RULES:
1. Every collection will automatically have an '_id' (ObjectId).
2. Every document MUST have `created_at` and `updated_at` (datetime).
3. Specify which fields should be indexed (e.g., Searchable terms, Foreign Keys).
4. Output ONLY the description. No markdown block.
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
