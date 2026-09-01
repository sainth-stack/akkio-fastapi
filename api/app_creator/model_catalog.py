"""Curated model catalog for App Builder chat + codegen."""

from __future__ import annotations

from typing import Any, Dict, List

# id must match OpenAI model slug passed to ChatOpenAI
APP_BUILDER_MODELS: List[Dict[str, Any]] = [
    {
        "id": "gpt-4o",
        "label": "GPT-4o",
        "description": "Best overall quality for PRD, architecture, and code",
        "tier": "flagship",
        "provider": "openai",
        "recommended": True,
    },
    {
        "id": "gpt-4o-mini",
        "label": "GPT-4o Mini",
        "description": "Fast and cost-efficient — great default for most apps",
        "tier": "balanced",
        "provider": "openai",
        "recommended": True,
    },
    {
        "id": "gpt-4.1",
        "label": "GPT-4.1",
        "description": "Latest GPT-4.1 with strong instruction following",
        "tier": "flagship",
        "provider": "openai",
    },
    {
        "id": "gpt-4.1-mini",
        "label": "GPT-4.1 Mini",
        "description": "Balanced speed and quality for planning steps",
        "tier": "balanced",
        "provider": "openai",
    },
    {
        "id": "gpt-4.1-nano",
        "label": "GPT-4.1 Nano",
        "description": "Ultra-fast iterations on UI copy and small edits",
        "tier": "fast",
        "provider": "openai",
    },
    {
        "id": "gpt-5",
        "label": "GPT-5",
        "description": "Newest flagship model for complex full-stack apps",
        "tier": "flagship",
        "provider": "openai",
    },
    {
        "id": "gpt-5-mini",
        "label": "GPT-5 Mini",
        "description": "Fast GPT-5 variant for codegen and planning",
        "tier": "balanced",
        "provider": "openai",
    },
    {
        "id": "gpt-5.2",
        "label": "GPT-5.2",
        "description": "Most capable — use for large PRDs and architecture",
        "tier": "flagship",
        "provider": "openai",
    },
    {
        "id": "gpt-4-turbo",
        "label": "GPT-4 Turbo",
        "description": "Proven turbo model with large context window",
        "tier": "balanced",
        "provider": "openai",
    },
    {
        "id": "gpt-4",
        "label": "GPT-4",
        "description": "Classic GPT-4 — reliable structured output",
        "tier": "legacy",
        "provider": "openai",
    },
    {
        "id": "gpt-3.5-turbo",
        "label": "GPT-3.5 Turbo",
        "description": "Budget-friendly for simple apps and prototypes",
        "tier": "fast",
        "provider": "openai",
    },
    {
        "id": "o1-preview",
        "label": "o1 Preview",
        "description": "Deep reasoning for complex architecture decisions",
        "tier": "reasoning",
        "provider": "openai",
    },
    {
        "id": "o1-mini",
        "label": "o1 Mini",
        "description": "Reasoning model with lower latency and cost",
        "tier": "reasoning",
        "provider": "openai",
    },
    {
        "id": "chatgpt-4o-latest",
        "label": "ChatGPT-4o Latest",
        "description": "Rolling latest ChatGPT-4o snapshot",
        "tier": "balanced",
        "provider": "openai",
    },
]

ALLOWED_MODELS = APP_BUILDER_MODELS

ALLOWED_MODEL_IDS = {m["id"] for m in APP_BUILDER_MODELS}

TIER_LABELS = {
    "flagship": "Flagship",
    "balanced": "Balanced",
    "fast": "Fast",
    "reasoning": "Reasoning",
    "legacy": "Legacy",
}


def normalize_model_choice(model_name: str | None, fallback: str = "gpt-4o-mini") -> str:
    """Return a valid catalog model id or fallback."""
    if model_name and model_name in ALLOWED_MODEL_IDS:
        return model_name
    return fallback if fallback in ALLOWED_MODEL_IDS else APP_BUILDER_MODELS[0]["id"]
