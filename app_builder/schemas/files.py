from pydantic import BaseModel
from typing import Dict, List, Optional

class GeneratedFiles(BaseModel):
    files: Dict[str, str]

class FileNode(BaseModel):
    path: str
    is_dir: bool
    children: Optional[List['FileNode']] = None

class FileContent(BaseModel):
    path: str
    content: str

class ErrorResponse(BaseModel):
    message: str
