"""
Pydantic schemas for the frontend-only planning pipeline.

Each planning stage produces:
  1. Markdown prose (streamed to the user)
  2. A validated JSON contract (consumed by code generators)

Contracts are stored together under app.plan_json:
  {
    "prd":           PRDPlan,
    "uiux":          UXPlan,
    "design_tokens": DesignTokenSchema,
    "blueprint":     Blueprint,
  }
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, validator

# ---------------------------------------------------------------------------
# Kit component allow-list (must match src/components/ui/index.ts)
# ---------------------------------------------------------------------------

KNOWN_KIT_COMPONENTS: frozenset = frozenset({
    "AppShell", "PageHeader", "Button", "Card", "StatCard",
    "DataTable", "FormField", "Modal", "ConfirmDialog", "Toast",
    "EmptyState", "LoadingState", "ErrorState", "Tabs",
    "LineChart", "BarChart", "PieChart", "StatusChip",
})

# Valid field types the deterministic generator understands
VALID_FIELD_TYPES: frozenset = frozenset({
    "string", "number", "boolean", "Date", "string[]", "number[]",
    "object", "enum",
})

# ---------------------------------------------------------------------------
# Stage 1 — PRD plan
# ---------------------------------------------------------------------------

class Feature(BaseModel):
    id: str       # "F001"
    name: str
    priority: Literal["must", "should", "could"]

    @validator("id")
    def id_format(cls, v: str) -> str:
        if not v.startswith("F"):
            raise ValueError(f"Feature id must start with 'F', got {v!r}")
        return v


class ScreenRef(BaseModel):
    id: str       # "S001"
    name: str     # "Contacts"
    purpose: str  # one-line description of what this screen does
    entities: List[str]   # entity names referenced on this screen

    @validator("id")
    def id_format(cls, v: str) -> str:
        if not v.startswith("S"):
            raise ValueError(f"Screen id must start with 'S', got {v!r}")
        return v


class PRDPlan(BaseModel):
    summary: str
    personas: List[str]
    goals: List[str]
    features: List[Feature]
    screens: List[ScreenRef]
    non_goals: List[str]

    @validator("screens")
    def at_least_one_screen(cls, v: List[ScreenRef]) -> List[ScreenRef]:
        if len(v) < 1:
            raise ValueError("At least 1 screen required in PRDPlan.screens")
        return v

    @validator("features")
    def at_least_one_feature(cls, v: List[Feature]) -> List[Feature]:
        if len(v) < 1:
            raise ValueError("At least 1 feature required in PRDPlan.features")
        return v


# ---------------------------------------------------------------------------
# Stage 2 — UX plan (per-screen detail)
# ---------------------------------------------------------------------------

class ScreenUX(BaseModel):
    screen_id: str        # matches PRDPlan.screens[*].id
    layout: str           # e.g. "2-col grid: sidebar nav + main content area"
    sections: List[Dict[str, str]]  # [{name, description}]
    components_from_kit: List[str]  # names from KNOWN_KIT_COMPONENTS
    states: Dict[str, str]          # {empty, loading, error} → short description
    actions: List[str]    # user actions ("Click row → detail modal", ...)
    navigation: List[str] # outgoing nav links ("Back to Dashboard", ...)

    @validator("components_from_kit", each_item=True)
    def kit_components_exist(cls, v: str) -> str:
        if v not in KNOWN_KIT_COMPONENTS:
            raise ValueError(
                f"Unknown kit component '{v}'. "
                f"Valid: {sorted(KNOWN_KIT_COMPONENTS)}"
            )
        return v


class UXPlan(BaseModel):
    screens: List[ScreenUX]

    @validator("screens")
    def at_least_one(cls, v: List[ScreenUX]) -> List[ScreenUX]:
        if len(v) < 1:
            raise ValueError("UXPlan.screens must have at least 1 entry")
        return v


# ---------------------------------------------------------------------------
# Stage 3 — Design tokens
# ---------------------------------------------------------------------------

class ColorPalette(BaseModel):
    primary: str
    primary_dark: str
    primary_light: str
    secondary: str
    background: str
    surface: str
    text: str
    muted: str
    border: str
    danger: str
    success: str
    warning: str
    info: str

    @validator("*", pre=True)
    def is_hex(cls, v: str) -> str:  # type: ignore[override]
        s = str(v).strip()
        if not (s.startswith("#") and len(s) in (4, 7, 9)):
            raise ValueError(f"Expected hex color, got {s!r}")
        return s


class DesignTokenSchema(BaseModel):
    light: ColorPalette
    dark: Optional[ColorPalette] = None
    font_family: str
    font_mono: str = "ui-monospace, 'Cascadia Code', monospace"
    border_radius: int = 10          # px
    chart_palette: List[str]         # 6–10 hex colors for charts


# ---------------------------------------------------------------------------
# Stage 4 — Blueprint (consumed by deterministic_generator.py)
# ---------------------------------------------------------------------------

class RouteSpec(BaseModel):
    path: str           # "/contacts"
    page: str           # "ContactsPage" (component name, no extension)
    screen_id: str      # links to PRDPlan.screens[*].id


class EntityField(BaseModel):
    name: str   # camelCase field name
    type: str   # one of VALID_FIELD_TYPES values

    @validator("type")
    def valid_type(cls, v: str) -> str:
        base = v.rstrip("[]").lower()
        if base not in {t.rstrip("[]").lower() for t in VALID_FIELD_TYPES}:
            raise ValueError(f"Unknown field type {v!r}")
        return v


class EntitySpec(BaseModel):
    name: str           # PascalCase singular: "Contact"
    fields: List[EntityField]

    @validator("fields")
    def has_id_field(cls, v: List[EntityField]) -> List[EntityField]:
        names = [f.name.lower() for f in v]
        if "id" not in names:
            raise ValueError("Entity must have an 'id' field")
        return v


class PageSpec(BaseModel):
    file: str                       # "ContactsPage.tsx"
    imports: List[str]              # relative import paths
    uses_entities: List[str]        # entity names (must exist in Blueprint.entities)
    uses_kit_components: List[str]  # must be in KNOWN_KIT_COMPONENTS

    @validator("file")
    def ends_in_tsx(cls, v: str) -> str:
        if not v.endswith(".tsx"):
            raise ValueError(f"Page file must end in .tsx, got {v!r}")
        return v

    @validator("uses_kit_components", each_item=True)
    def kit_component_exists(cls, v: str) -> str:
        if v not in KNOWN_KIT_COMPONENTS:
            raise ValueError(f"Unknown kit component '{v}'")
        return v


class MockRowHint(BaseModel):
    entity: str                         # entity name
    count: int                          # rows to generate (1–50)
    value_hints: Dict[str, str]         # field → realistic example description

    @validator("count")
    def sane_count(cls, v: int) -> int:
        if not (1 <= v <= 50):
            raise ValueError(f"count must be 1–50, got {v}")
        return v


class NavItemSpec(BaseModel):
    label: str   # sidebar label
    path: str    # router path
    icon: str    # MUI icon name (e.g. "Dashboard", "People")


class Blueprint(BaseModel):
    routes: List[RouteSpec]
    pages: List[PageSpec]
    entities: List[EntitySpec]
    mock_plan: List[MockRowHint]
    nav_items: List[NavItemSpec]
    file_order: List[str]   # preferred generation order for code agents

    @validator("routes")
    def at_least_one_route(cls, v: List[RouteSpec]) -> List[RouteSpec]:
        if len(v) < 1:
            raise ValueError("Blueprint.routes must have at least 1 entry")
        return v


# ---------------------------------------------------------------------------
# Aggregate plan_json envelope (stored in DB)
# ---------------------------------------------------------------------------

class PlanJson(BaseModel):
    prd: Optional[PRDPlan] = None
    uiux: Optional[UXPlan] = None
    design_tokens: Optional[DesignTokenSchema] = None
    blueprint: Optional[Blueprint] = None


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_blueprint(
    blueprint: Blueprint,
    prd: Optional[PRDPlan] = None,
) -> List[str]:
    """
    Validate internal consistency of a Blueprint (and optionally against a PRDPlan).

    Returns a list of error strings (empty = valid).
    """
    errors: List[str] = []

    # 1. Every screen_id in routes references a known PRD screen (if prd provided)
    if prd is not None:
        prd_screen_ids = {s.id for s in prd.screens}
        for route in blueprint.routes:
            if route.screen_id and route.screen_id not in prd_screen_ids:
                errors.append(
                    f"Route '{route.path}' references unknown screen_id '{route.screen_id}'. "
                    f"Known: {sorted(prd_screen_ids)}"
                )

    # 2. Every route has a matching page file
    route_pages = {r.page for r in blueprint.routes}
    page_component_names = {p.file.replace(".tsx", "") for p in blueprint.pages}
    for comp in route_pages:
        if comp not in page_component_names:
            errors.append(
                f"Route points to page component '{comp}' "
                f"but no matching .tsx file found in pages list."
            )

    # 3. Every entity referenced by a page exists in Blueprint.entities
    entity_names = {e.name for e in blueprint.entities}
    for page in blueprint.pages:
        for entity in page.uses_entities:
            if entity not in entity_names:
                errors.append(
                    f"Page '{page.file}' uses entity '{entity}' "
                    f"which is not declared in Blueprint.entities. "
                    f"Known: {sorted(entity_names)}"
                )

    # 4. Every kit component referenced by a page is in the known kit
    for page in blueprint.pages:
        for comp in page.uses_kit_components:
            if comp not in KNOWN_KIT_COMPONENTS:
                errors.append(
                    f"Page '{page.file}' references unknown kit component '{comp}'. "
                    f"Valid: {sorted(KNOWN_KIT_COMPONENTS)}"
                )

    # 5. No duplicate route paths
    seen_paths: set = set()
    for route in blueprint.routes:
        if route.path in seen_paths:
            errors.append(f"Duplicate route path: '{route.path}'")
        seen_paths.add(route.path)

    # 6. Every mock_plan entity exists
    for mock in blueprint.mock_plan:
        if mock.entity not in entity_names:
            errors.append(
                f"mock_plan references entity '{mock.entity}' "
                f"which is not in Blueprint.entities."
            )

    return errors


def repair_blueprint_prompt(blueprint_dict: Dict[str, Any], errors: List[str]) -> str:
    """Return a prompt snippet asking the LLM to fix the Blueprint."""
    error_list = "\n".join(f"  - {e}" for e in errors)
    return (
        f"The Blueprint you returned has validation errors:\n{error_list}\n\n"
        f"Current Blueprint:\n```json\n{blueprint_dict}\n```\n\n"
        "Return ONLY the corrected Blueprint JSON (same schema, no markdown wrapper)."
    )
