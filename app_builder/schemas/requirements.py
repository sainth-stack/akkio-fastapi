from pydantic import BaseModel

class UserRequirement(BaseModel):
    description: str
