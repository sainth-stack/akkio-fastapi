"""Per-model token pricing (USD per 1M tokens) for LLM observability cost display."""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

# Approximate public list pricing (USD per 1M tokens). Unknown models fall back by tier.
_MODEL_PRICE_PER_M: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5": (5.00, 15.00),
    "gpt-5-mini": (0.50, 1.50),
    "gpt-5.2": (6.00, 18.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1-preview": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "chatgpt-4o-latest": (2.50, 10.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.25, 1.25),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
}

_TIER_FALLBACK: Tuple[float, float] = (1.00, 3.00)
_FAST_FALLBACK: Tuple[float, float] = (0.20, 0.80)
_REASONING_FALLBACK: Tuple[float, float] = (8.00, 24.00)


def normalize_model_id(model: str) -> str:
    m = (model or "").strip().lower()
    m = re.sub(r"^models/", "", m)
    return m.split(":")[0]


def _resolve_rates(model: str) -> Tuple[float, float]:
    mid = normalize_model_id(model)
    if mid in _MODEL_PRICE_PER_M:
        return _MODEL_PRICE_PER_M[mid]
    for key, rates in _MODEL_PRICE_PER_M.items():
        if mid.startswith(key) or key in mid:
            return rates
    if re.search(r"mini|nano|flash|haiku|3\.5", mid):
        return _FAST_FALLBACK
    if re.search(r"o1|reasoning", mid):
        return _REASONING_FALLBACK
    return _TIER_FALLBACK


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _resolve_rates(model)
    cost = (max(0, input_tokens) / 1_000_000.0) * in_rate
    cost += (max(0, output_tokens) / 1_000_000.0) * out_rate
    return round(cost, 6)


def extract_token_breakdown(obj: Any) -> Dict[str, int]:
    from api.akkio.usage_tracking import extract_total_tokens

    total = extract_total_tokens(obj)
    input_tokens = 0
    output_tokens = 0

    usage_md = getattr(obj, "usage_metadata", None)
    if isinstance(usage_md, dict):
        it = usage_md.get("input_tokens")
        ot = usage_md.get("output_tokens")
        if isinstance(it, int):
            input_tokens = it
        if isinstance(ot, int):
            output_tokens = ot

    resp_md = getattr(obj, "response_metadata", None)
    if isinstance(resp_md, dict):
        token_usage = resp_md.get("token_usage") or resp_md.get("usage")
        if isinstance(token_usage, dict):
            if isinstance(token_usage.get("input_tokens"), int):
                input_tokens = token_usage["input_tokens"]
            if isinstance(token_usage.get("output_tokens"), int):
                output_tokens = token_usage["output_tokens"]
            if isinstance(token_usage.get("prompt_tokens"), int) and not input_tokens:
                input_tokens = token_usage["prompt_tokens"]
            if isinstance(token_usage.get("completion_tokens"), int) and not output_tokens:
                output_tokens = token_usage["completion_tokens"]

    usage = getattr(obj, "usage", None)
    if usage is not None:
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int):
                input_tokens = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int):
                output_tokens = usage["completion_tokens"]
        else:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
            if isinstance(pt, int):
                input_tokens = pt
            if isinstance(ct, int):
                output_tokens = ct

    if total <= 0 and (input_tokens or output_tokens):
        total = input_tokens + output_tokens
    if total > 0 and input_tokens == 0 and output_tokens == 0:
        input_tokens = max(0, int(total * 0.75))
        output_tokens = max(0, total - input_tokens)

    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total or (input_tokens + output_tokens)),
    }
