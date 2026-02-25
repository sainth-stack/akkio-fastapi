import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

async def backend_generator_agent(
    structured_requirement: Dict[str, Any],
    blueprint: Dict[str, Any],
    api_contract: Dict[str, Any],
    db_schema: str,
    llm=None
) -> Dict[str, str]:
    """
    Backend Code Generator Agent - Generates FastAPI app files using SQLite + SQLAlchemy.
    """
    if llm is None:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0.2)

    system_prompt = """You are a Senior Backend Developer. Generate FastAPI code using SQLite and SQLAlchemy.

CRITICAL RULES:
1. Use SQLAlchemy + SQLite. No external database. DATABASE_URL defaults to sqlite:///./app.db
2. Use integer primary keys for all tables.
3. Pydantic v2: model_config = ConfigDict(from_attributes=True), .model_dump()
4. backend/database.py: create_engine, SessionLocal, Base (declarative_base)
5. backend/models.py: SQLAlchemy model classes with Column, Integer, String, Boolean, DateTime, Text
6. backend/schemas.py: Pydantic BaseModel classes (TodoCreate, Todo, etc)
7. backend/main.py: FastAPI app, Depends(get_db), models.Base.metadata.create_all(bind=database.engine) on startup
8. requirements.txt: fastapi, uvicorn, sqlalchemy, pydantic (no version numbers)
9. CORS middleware allowing all origins
10. All endpoints use Depends(database.get_db) for Session

Output ONLY a JSON mapping: {"backend/main.py": "...", "backend/models.py": "...", "backend/schemas.py": "...", "backend/database.py": "...", "backend/requirements.txt": "..."}"""

    context = {
        "structured_requirement": structured_requirement,
        "blueprint": blueprint,
        "api_contract": api_contract,
        "db_schema": db_schema
    }
    user_prompt = f"Context: {json.dumps(context, indent=2)}"

    response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}
