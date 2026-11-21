"""
Shared helper for extracting validated Pydantic models from LLM responses.

Used by the frontend-only planning agents to turn markdown output into
structured JSON contracts with up to N retry attempts on validation failure.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("app_builder")

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# JSON extraction from LLM text
# ---------------------------------------------------------------------------

def extract_json_block(text: str) -> Optional[str]:
    """
    Pull the first valid JSON object out of an LLM response.

    Tries (in order):
      1. ```json … ``` fenced block
      2. ``` … ``` fenced block
      3. First { … } balanced block in the raw text
    """
    # 1. ```json block
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2. plain ``` block
    m = re.search(r"```\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3. Balanced { … } scan
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Retry-aware JSON extractor
# ---------------------------------------------------------------------------

async def extract_validated(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    max_retries: int = 2,
    context_label: str = "",
) -> Optional[T]:
    """
    Call the LLM with (system_prompt, user_prompt), parse the JSON response,
    validate with `schema`, and retry up to `max_retries` times on any error.

    Returns:
        Validated Pydantic model instance, or None if all attempts fail.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    last_error: str = ""
    current_user_prompt = user_prompt

    for attempt in range(max_retries + 1):
        label = f"{context_label} attempt {attempt + 1}/{max_retries + 1}"
        try:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=current_user_prompt),
            ])
            content: str = response.content if hasattr(response, "content") else str(response)

            json_str = extract_json_block(content)
            if not json_str:
                raise ValueError("No JSON object found in LLM response")

            data: Dict[str, Any] = json.loads(json_str)
            model_instance = schema(**data)
            logger.info("[structured_output] %s validated OK", label)
            return model_instance

        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            last_error = str(exc)
            logger.warning("[structured_output] %s failed: %s", label, last_error[:200])
            if attempt < max_retries:
                current_user_prompt = (
                    f"{user_prompt}\n\n"
                    f"---\n"
                    f"Your previous response was invalid. Error: {last_error}\n"
                    f"Return ONLY valid JSON matching the schema. No markdown, no explanation."
                )

    logger.error(
        "[structured_output] %s exhausted retries. Last error: %s",
        context_label,
        last_error,
    )
    return None


# ---------------------------------------------------------------------------
# Utility: cheap non-streaming LLM call (wraps async)
# ---------------------------------------------------------------------------

async def call_llm_once(llm: Any, system: str, user: str) -> str:
    """Invoke LLM and return content string. Does NOT stream."""
    from langchain_core.messages import HumanMessage, SystemMessage

    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content if hasattr(response, "content") else str(response)
