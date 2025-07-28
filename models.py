from pydantic import BaseModel
from typing import Optional, Dict, Any


class DBConnectionRequest(BaseModel):
    username: str
    password: str
    database: str
    host: str
    port: str


# Pydantic model for request
class GenAIBotRequest(BaseModel):
    prompt: str

class ModelRequest(BaseModel):
    model: str
    col: str
    frequency: Optional[str] = None
    tenure: Optional[int] = None
