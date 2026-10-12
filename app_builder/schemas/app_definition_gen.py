"""LLM output schema for AppDefinition (mock/data.json) generation."""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class NavItemGen(BaseModel):
    label: str
    path: str
    icon: str = "Dashboard"
    pageId: str


class PageGen(BaseModel):
    id: str
    title: str
    subtitle: str = ""
    blocks: List[Dict[str, Any]] = Field(default_factory=list)


class AppDefinitionGen(BaseModel):
    appName: str = Field(min_length=1, max_length=120)
    description: str = ""
    domain: str = "generic"
    navigation: List[NavItemGen] = Field(min_length=1, max_length=12)
    pages: List[PageGen] = Field(min_length=1, max_length=12)
    mock: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("navigation", "pages")
    @classmethod
    def non_empty(cls, v: List) -> List:
        if not v:
            raise ValueError("At least one entry required")
        return v
