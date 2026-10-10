"""AppDefinition — single JSON document rendered by base-frontend-vite-mui."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class AppMeta(BaseModel):
    appName: str
    description: str = ""
    domain: str = "generic"


class AppTheme(BaseModel):
    primary: str = "#1976d2"
    primaryDark: str = "#1565c0"
    primaryLight: str = "#42a5f5"
    secondary: str = "#9c27b0"
    background: str = "#f5f7fb"
    surface: str = "#ffffff"
    text: str = "#0f172a"
    muted: str = "#64748b"
    danger: str = "#d32f2f"
    success: str = "#2e7d32"
    warning: str = "#ed6c02"
    info: str = "#0288d1"
    fontFamily: str = "Inter"
    chartPalette: List[str] = Field(
        default_factory=lambda: ["#1976d2", "#2e7d32", "#ed6c02", "#9c27b0", "#0288d1", "#d32f2f"]
    )


class NavItemDef(BaseModel):
    label: str
    path: str
    icon: str = "Dashboard"
    pageId: str


class KpiItemDef(BaseModel):
    label: str
    valueKey: str
    color: Literal["primary", "success", "error", "warning", "info", "default"] = "primary"


class KpiRowBlock(BaseModel):
    type: Literal["kpiRow"] = "kpiRow"
    items: List[KpiItemDef]


class ChartSeriesDef(BaseModel):
    dataKey: str
    name: Optional[str] = None
    color: Optional[str] = None


class LineChartBlock(BaseModel):
    type: Literal["lineChart"] = "lineChart"
    title: str = ""
    dataKey: str
    xAxisKey: str = "name"
    series: List[ChartSeriesDef]
    height: int = 280


class BarChartBlock(BaseModel):
    type: Literal["barChart"] = "barChart"
    title: str = ""
    dataKey: str
    xAxisKey: str = "name"
    series: List[ChartSeriesDef]
    height: int = 280


class PieChartBlock(BaseModel):
    type: Literal["pieChart"] = "pieChart"
    title: str = ""
    dataKey: str
    height: int = 280


class TableColumnDef(BaseModel):
    field: str
    header: str
    align: Optional[Literal["left", "center", "right"]] = None


class DataTableBlock(BaseModel):
    type: Literal["dataTable"] = "dataTable"
    title: str = ""
    dataKey: str
    columns: List[TableColumnDef]
    searchable: bool = True
    emptyMessage: str = "No records found"


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    content: str


BlockDef = Union[
    KpiRowBlock,
    LineChartBlock,
    BarChartBlock,
    PieChartBlock,
    DataTableBlock,
    TextBlock,
]


class PageDef(BaseModel):
    id: str
    title: str
    subtitle: str = ""
    blocks: List[Dict[str, Any]] = Field(default_factory=list)


class AppDefinition(BaseModel):
    version: int = 1
    meta: AppMeta
    theme: AppTheme = Field(default_factory=AppTheme)
    navigation: List[NavItemDef]
    pages: List[PageDef]
    mock: Dict[str, Any] = Field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")
