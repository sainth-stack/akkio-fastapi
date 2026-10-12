"""Canonical Agentic Builder agent IDs and planning-step mapping."""
from __future__ import annotations

from typing import Dict, Tuple

# Primary agents shown in the UI pipeline
AGENT_PLANNER = "planner_agent"
AGENT_STYLING = "styling_agent"
AGENT_CODING = "coding_agent"
AGENT_INTEGRATION = "integration_agent"
AGENT_VALIDATOR = "validator_agent"
AGENT_APP_DEFINITION = "app_definition_agent"

PLANNING_STEP_AGENT: Dict[str, Tuple[str, str]] = {
    "prd": (AGENT_PLANNER, "Analyzing product intent and drafting production PRD…"),
    "uiux": (AGENT_PLANNER, "Designing screen flows, navigation IA, and UX patterns…"),
    "style": (AGENT_STYLING, "Generating design system — color, type, spacing, components…"),
    "architecture": (AGENT_PLANNER, "Defining frontend architecture, routes, and data contracts…"),
    "plan": (AGENT_PLANNER, "Creating implementation plan for codegen agents…"),
}

PLANNING_STEP_COMPLETE: Dict[str, str] = {
    "prd": "PRD ready — features, personas, and acceptance criteria captured.",
    "uiux": "UI/UX ready — screens and flows defined for App Definition agent.",
    "style": "Design tokens ready — brand palette applied for production shell.",
    "architecture": "Architecture ready — routes and JSON contracts locked for codegen.",
    "plan": "Implementation plan ready — safe to run Generate Code.",
}
