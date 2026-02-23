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
    Backend Code Generator Agent - Generates FastAPI app files.
    Strictly follows the API contract and DB schema.
    """
    if llm is None:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0.2)

    system_prompt = """You are a Senior Backend Developer. Your task is to generate robust FastAPI code using MongoDB.
You MUST strictly follow the provided API contract and DB schema.

MONGODB CONNECTION STRING (hardcoded):
mongodb+srv://prashanth:BnHRQrqZHdnosfEe@cluster0.cpydc.mongodb.net/akkio?retryWrites=true&w=majority

PYTHON 3.13 COMPATIBILITY — CRITICAL:
All generated code MUST run on Python 3.13+. Follow these rules strictly.

CRITICAL TECHNICAL RULES:
1. Use PyMongo (SYNCHRONOUS driver) — NOT Motor, NOT SQLAlchemy, NOT SQLite.
   Motor uses deprecated asyncio.coroutine which is REMOVED in Python 3.13.
2. Use `pymongo.MongoClient` for the database client. All DB calls are SYNCHRONOUS.
3. Documents use `_id: ObjectId` from `bson`. Expose as string `id` in responses.
4. Every document MUST have `created_at` and `updated_at` fields (datetime, set at insert time).
5. Pydantic v2 models MUST:
   - Use `from pydantic import BaseModel, ConfigDict, field_validator`.
   - Use `model_config = ConfigDict(arbitrary_types_allowed=True)` at top.
   - Have a helper to convert ObjectId to str: `@field_validator("id", mode="before") @classmethod def coerce_id(cls, v): return str(v)`.
   - Use `Optional[str] = None` for `id` (assigned by MongoDB).
   - NEVER use `class Config:` with `orm_mode`. Use `model_config = ConfigDict(from_attributes=True)`.
   - Use `.model_dump()` instead of `.dict()`.
6. `backend/database.py`: Holds `pymongo.MongoClient`, `get_database()` function (sync, not async).
7. `backend/main.py`:
   - On startup: connect to MongoDB, create indexes if needed.
   - NEVER use `database.Base.metadata.create_all` or any SQLAlchemy/SQLite code.
   - All DB operations are SYNCHRONOUS: `collection.find_one(...)`, `collection.insert_one(...)`, etc.
   - Use `list(collection.find({}).limit(100))` for list endpoints.
   - Endpoints are normal `def`, NOT `async def`.
8. Include CORS middleware allowing all origins.
9. requirements.txt MUST include ONLY package names — NO version numbers (NO ==, >=, ~=, <).
   Required packages: fastapi, uvicorn, pymongo, pydantic

FILE LAYOUT — STRICT SEPARATION:
- backend/models.py:  ONLY MongoDB document helpers (serialization, ObjectId conversion). NO Pydantic schemas. NO SQLAlchemy Base.
- backend/schemas.py: ALL Pydantic BaseModel classes (e.g. TodoCreate, TodoUpdate, TodoResponse). main.py imports schemas from HERE.
- backend/database.py: PyMongo MongoClient + get_database().
- backend/main.py: FastAPI app, endpoints, CORS. Imports: `from models import ...` for DB helpers, `from schemas import ...` for Pydantic models.
- NEVER put Pydantic schemas in models.py. NEVER import TodoCreate/TodoUpdate from models.

Output ONLY a JSON mapping of filenames to file content:
{
  "backend/main.py": "content",
  "backend/models.py": "content",
  "backend/schemas.py": "content",
  "backend/database.py": "content",
  "backend/requirements.txt": "content"
}

CRITICAL RULES:
1. All endpoints in main.py must match the API CONTRACT.
2. Output ONLY the raw JSON string. Do NOT include markdown blocks.
3. If the code is long, do NOT truncate. Every file listed above MUST be present and complete.
4. NEVER use Motor, SQLAlchemy, SQLite, or any async MongoDB driver.
5. NEVER use Pydantic v1 patterns (orm_mode, .dict(), class Config).
6. NEVER include `Base.metadata.create_all` or `SessionLocal` - those are for SQL, not MongoDB.
"""

    context = {
        "structured_requirement": structured_requirement,
        "blueprint": blueprint,
        "api_contract": api_contract,
        "db_schema": db_schema
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
        return {}
