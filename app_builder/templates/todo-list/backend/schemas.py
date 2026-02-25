from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class TodoBase(BaseModel):
    title: str
    description: Optional[str] = None
    completed: bool = False
    priority: str = "medium"


class TodoCreate(TodoBase):
    pass


class Todo(TodoBase):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)
