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

def response_text(content: Any) -> str:
    """Flatten a chat response into text. Some models return a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _balanced_span(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_block(text: str) -> Optional[str]:
    """Pull the first JSON object or array out of an LLM response."""
    text = response_text(text)
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.DOTALL)
    scopes = [fenced.group(1)] if fenced else []
    scopes.append(text)
    for scope in scopes:
        if scope.strip().startswith("["):
            array = _balanced_span(scope, "[", "]")
            if array:
                return array
        obj = _balanced_span(scope, "{", "}")
        if obj:
            return obj
        array = _balanced_span(scope, "[", "]")
        if array:
            return array
    return None


def loads_json_object(text: str) -> Any:
    """Parse model JSON, including trailing commas that strict json.loads rejects."""
    raw = extract_json_block(text)
    if not raw:
        raise ValueError("No JSON object found in LLM response")
    relaxed = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(relaxed)
    except json.JSONDecodeError:
        return json.loads(raw)


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
            content = response.content if hasattr(response, "content") else str(response)

            data = loads_json_object(response_text(content))
            if isinstance(data, list):
                fields = set(getattr(schema, "model_fields", {}))
                if "blocks" in fields:
                    data = {"blocks": data}
                elif "pages" in fields or "navigation" in fields:
                    data = {"pages": data}
                else:
                    raise ValueError("JSON array is not an object")
            if not isinstance(data, dict):
                raise ValueError("JSON response must be an object")
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
