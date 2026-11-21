"""Multi-phase LLM schemas for AppDefinition generation."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator


class NavItemPlan(BaseModel):
    label: str
    path: str
    icon: str = "Dashboard"
    pageId: str


class PageOutlinePlan(BaseModel):
    id: str
    title: str
    subtitle: str = ""


def _shell_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return slug or "page"


def _as_dict_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            if isinstance(item, dict):
                rows.append({"id": key, **item})
            else:
                rows.append({"id": key, "title": item})
        return rows
    return []


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

    @model_validator(mode="before")
    @classmethod
    def normalize_shell(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        for key in ("appShell", "shell", "plan", "AppShellPlan", "result"):
            inner = value.get(key)
            if isinstance(inner, dict) and any(name in inner for name in ("pages", "navigation", "appName", "app_name", "screens")):
                value = inner
                break
        app_name = str(value.get("appName") or value.get("app_name") or value.get("name") or "Workspace").strip() or "Workspace"
        pages = []
        for index, page in enumerate(_as_dict_list(value.get("pages") or value.get("screens"))):
            if isinstance(page, str):
                pages.append({"id": _shell_slug(page), "title": page, "subtitle": ""})
                continue
            if not isinstance(page, dict):
                continue
            title = str(page.get("title") or page.get("name") or page.get("label") or f"Page {index + 1}")
            page_id = str(page.get("id") or page.get("pageId") or page.get("page_id") or _shell_slug(title))
            pages.append({"id": page_id, "title": title, "subtitle": str(page.get("subtitle") or page.get("description") or "")})
        navigation = []
        for index, item in enumerate(_as_dict_list(value.get("navigation") or value.get("nav") or value.get("routes"))):
            if isinstance(item, str):
                page_id = pages[index]["id"] if index < len(pages) else _shell_slug(item)
                navigation.append({"label": item, "path": "/" if index == 0 else f"/{page_id}", "icon": "Article", "pageId": page_id})
                continue
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("title") or item.get("name") or (pages[index]["title"] if index < len(pages) else f"Page {index + 1}"))
            page_id = str(item.get("pageId") or item.get("page_id") or item.get("id") or (pages[index]["id"] if index < len(pages) else _shell_slug(label)))
            path = str(item.get("path") or item.get("route") or item.get("href") or ("/" if index == 0 else f"/{page_id}"))
            navigation.append({"label": label, "path": path, "icon": str(item.get("icon") or "Article"), "pageId": page_id})
        if pages and not navigation:
            navigation = [
                {"label": page["title"], "path": "/" if index == 0 else f"/{page['id']}", "icon": "Article", "pageId": page["id"]}
                for index, page in enumerate(pages)
            ]
        if navigation and not pages:
            pages = [{"id": item["pageId"], "title": item["label"], "subtitle": ""} for item in navigation]
        if len(navigation) < len(pages):
            for index in range(len(navigation), len(pages)):
                page = pages[index]
                navigation.append({"label": page["title"], "path": f"/{page['id']}", "icon": "Article", "pageId": page["id"]})
        elif len(pages) < len(navigation):
            for item in navigation[len(pages):]:
                pages.append({"id": item["pageId"], "title": item["label"], "subtitle": ""})
        seen_paths = set()
        seen_ids = set()
        for index, item in enumerate(navigation):
            page_id = item["pageId"]
            if page_id in seen_ids:
                page_id = f"{page_id}_{index + 1}"
                item["pageId"] = page_id
            seen_ids.add(page_id)
            if index < len(pages):
                pages[index]["id"] = page_id
            path = "/" if index == 0 else item["path"]
            if index and (path == "/" or path in seen_paths):
                path = f"/{page_id}"
            item["path"] = path
            seen_paths.add(path)
        if not navigation or not pages:
            return value
        return {
            "appName": app_name[:120].rstrip(),
            "description": str(value.get("description") or ""),
            "domain": str(value.get("domain") or "generic") or "generic",
            "navigation": navigation[:12],
            "pages": pages[:12],
        }


class PageBlocksPlan(BaseModel):
    blocks: List[Dict[str, Any]] = Field(min_length=1, max_length=24)

    @field_validator("blocks")
    @classmethod
    def valid_blocks(cls, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from app_builder.services.definition_contract import validate_blocks

        return validate_blocks(blocks)


class PageDataPlan(BaseModel):
    data: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def nest_datasets(cls, value: Any) -> Any:
        from app_builder.services.definition_contract import coerce_page_data

        if not isinstance(value, dict):
            return value
        return {"data": coerce_page_data(value)}
