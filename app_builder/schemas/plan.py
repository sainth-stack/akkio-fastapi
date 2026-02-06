from pydantic import BaseModel
from typing import List

class ProjectPlan(BaseModel):
    steps: List[str]
