from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime


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


# Multi-Model Training Models
class MultiModelTrainingSession(BaseModel):
    session_id: str
    model_name: str
    user_email: str
    system_prompt: str
    files: List[Dict[str, Any]]  # [{filename, file_type, storage_path}]
    status: str  # 'pending', 'training', 'completed', 'failed'
    progress: int  # 0-100
    stage: str  # Current training stage
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None


class MultiModelQuery(BaseModel):
    model_name: str
    user_email: str
    query: str
    session_id: Optional[str] = None


class MultiModelResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]  # [{file_name, file_type, relevance, excerpt}]
    reasoning: str
    agents_used: List[str]
