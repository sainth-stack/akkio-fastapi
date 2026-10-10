"""AppSpec v1 — JSON-only applications rendered by the shared spec renderer."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ThemeSpec(BaseModel):
    primary: str = "#1565C0"
    primaryDark: str = "#0D47A1"
    secondary: str = "#7B1FA2"
    background: str = "#F5F5F5"
    surface: str = "#FFFFFF"
    text: str = "#212121"
    muted: str = "#757575"


class BrandingSpec(BaseModel):
    appName: str = "Application"
    headerTitle: str = "Dashboard"
    tagline: str = ""


class NavItemSpec(BaseModel):
    id: str
    label: str
    icon: str = "Dashboard"
    route: str
    screenId: str


class KpiSpec(BaseModel):
    id: str
    label: str
    value: str | int | float
    tone: str = "primary"  # primary | success | warning | error
    icon: str = "Dashboard"


class ChartSeriesSpec(BaseModel):
    dataKey: str
    name: str = ""
    color: str = "#1565C0"


class ChartSpec(BaseModel):
    id: str
    title: str
    type: Literal["line", "bar"] = "line"
    xAxisKey: str = "label"
    series: List[ChartSeriesSpec] = Field(default_factory=list)
    data: List[Dict[str, Any]] = Field(default_factory=list)


class TableColumnSpec(BaseModel):
    key: str
    label: str
    align: Literal["left", "right", "center"] = "left"


class ScreenSpec(BaseModel):
    id: str
    type: Literal["dashboard", "table", "alerts", "reports", "detail"] = "table"
    title: str
    subtitle: str = ""
    entityId: Optional[str] = None
    kpis: List[KpiSpec] = Field(default_factory=list)
    charts: List[ChartSpec] = Field(default_factory=list)
    columns: List[TableColumnSpec] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    reportSummary: List[KpiSpec] = Field(default_factory=list)
    detailFields: List[Dict[str, str]] = Field(default_factory=list)


class EntityFieldSpec(BaseModel):
    key: str
    label: str
    type: str = "string"


class EntitySpec(BaseModel):
    id: str
    label: str
    fields: List[EntityFieldSpec] = Field(default_factory=list)


class AppSpecV1(BaseModel):
    version: Literal[1] = 1
    meta: Dict[str, Any] = Field(default_factory=dict)
    branding: BrandingSpec = Field(default_factory=BrandingSpec)
    theme: ThemeSpec = Field(default_factory=ThemeSpec)
    navigation: List[NavItemSpec] = Field(default_factory=list)
    screens: List[ScreenSpec] = Field(default_factory=list)
    entities: List[EntitySpec] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


def validate_app_spec(data: Dict[str, Any]) -> AppSpecV1:
    return AppSpecV1.model_validate(data)
