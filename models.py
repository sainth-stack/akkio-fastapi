from pydantic import BaseModel
from typing import Optional

class DBConnectionRequest(BaseModel):
    username: str
    password: str
    database: str
    host: str
    port: str