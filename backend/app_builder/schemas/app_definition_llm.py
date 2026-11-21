"""Multi-phase LLM schemas for AppDefinition generation."""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class NavItemPlan(BaseModel):
    label: str
    path: str
    icon: str = "Dashboard"
    pageId: str


class PageOutlinePlan(BaseModel):
    id: str
    title: str
    subtitle: str = ""


class AppShellPlan(BaseModel):
    appName: str = Field(min_length=1, max_length=120)
    description: str = ""
    domain: str = "generic"
    navigation: List[NavItemPlan] = Field(min_length=1, max_length=12)
    pages: List[PageOutlinePlan] = Field(min_length=1, max_length=12)

    @field_validator("navigation", "pages")
    @classmethod
    def non_empty(cls, v: List) -> List:
        if not v:
            raise ValueError("At least one entry required")
        return v


class PageBlocksPlan(BaseModel):
    blocks: List[Dict[str, Any]] = Field(min_length=2, max_length=24)


class PageDataPlan(BaseModel):
    data: Dict[str, Any] = Field(default_factory=dict)
