import os
import re
from typing import Tuple, Any, Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add parent directory to path to import llm_config
sys.path.append(str(Path(__file__).resolve().parents[3]))
from llm_config import get_api_key

# Ensure .env variables (like OPENAI_API_KEY) are loaded in local/dev
load_dotenv()


def get_openai_client(user_email: Optional[str] = None):
    """Get OpenAI client with lazy initialization using llm_config"""
    api_key = get_api_key(user_email)
    return OpenAI(api_key=api_key)


def detect_arabic_language(text: str) -> bool:
    """Detect if the text contains Arabic characters"""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    return bool(arabic_pattern.search(text))


def get_language_context(query: str) -> Tuple[str, str]:
    """
    Detect language and return appropriate context and instructions
    Returns: (language, language_instructions)
    """
    is_arabic = detect_arabic_language(query)
    if is_arabic:
        return "arabic", """
        LANGUAGE REQUIREMENTS FOR ARABIC:
        - Respond entirely in Arabic
        - Use proper Arabic RTL (Right-to-Left) text direction
        - Use Arabic HTML structure with dir="rtl" attribute
        - Maintain professional Arabic terminology
        - Use proper Arabic formatting and punctuation
        """
    else:
        return "english", """
        LANGUAGE REQUIREMENTS FOR ENGLISH:
        - Respond in clear, professional English
        - Use proper HTML formatting
        - Maintain professional terminology
        """


def call_llm_with_usage(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: Optional[float] = 0.3,
    email: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Common function for LLM calls with automatic credit deduction.
    
    1. Calls OpenAI ChatCompletion.
    2. Records usage (deducts credits) via record_llm_usage.
    3. Returns the full response object.
    
    Args:
        messages: List of message dicts (role, content)
        model: Model name (if None, uses user's preferred model from llm_config)
        temperature: Temperature
        email: User email for credit deduction and model lookup
        **kwargs: Additional OpenAI args (e.g., max_tokens, response_format)
        
    Returns:
        OpenAI response object
    """
    # Import here to avoid circular dependencies if any
    # Assuming record_llm_usage is in ..usage_tracking (which is api/akkio/usage_tracking.py)
    # Since llm.py is in api/akkio/explore_functions, parent is api/akkio
    from ..usage_tracking import record_llm_usage
    from llm_config import get_model_name
    
    # Get model from user config if not specified
    if model is None:
        model = get_model_name(email)
    
    client = get_openai_client(email)
    request_args = {
        "model": model,
        "messages": messages,
        **kwargs,
    }
    # Some models only support provider-default temperature and reject explicit values.
    if temperature is not None:
        request_args["temperature"] = temperature

    try:
        response = client.chat.completions.create(**request_args)
    except Exception as e:
        # Retry without temperature if model doesn't support custom values (e.g. o1, reasoning models)
        err_msg = str(e).lower()
        if temperature is not None and ("temperature" in err_msg and ("unsupported" in err_msg or "does not support" in err_msg)):
            request_args.pop("temperature", None)
            response = client.chat.completions.create(**request_args)
        else:
            raise e

    # Track usage
    record_llm_usage(email, response)
    return response


async def stream_llm_with_usage(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: Optional[float] = 0.3,
    email: Optional[str] = None,
    **kwargs
):
    """
    Streaming version of LLM call that yields chunks as they arrive.
    
    Args:
        messages: List of message dicts (role, content)
        model: Model name (if None, uses user's preferred model from llm_config)
        temperature: Temperature
        email: User email for credit deduction and model lookup
        **kwargs: Additional OpenAI args
        
    Yields:
        str: Text chunks from the streaming response
    """
    from ..usage_tracking import record_llm_usage
    from llm_config import get_model_name
    
    # Get model from user config if not specified
    if model is None:
        model = get_model_name(email)
    
    client = get_openai_client(email)
    full_content = ""
    full_response = None

    request_args = {
        "model": model,
        "messages": messages,
        "stream": True,
        **kwargs,
    }
    if temperature is not None:
        request_args["temperature"] = temperature

    try:
        stream = client.chat.completions.create(**request_args)
    except Exception as e:
        # Retry without temperature if model doesn't support custom values
        err_msg = str(e).lower()
        if temperature is not None and ("temperature" in err_msg and ("unsupported" in err_msg or "does not support" in err_msg)):
            request_args.pop("temperature", None)
            stream = client.chat.completions.create(**request_args)
        else:
            raise e

    for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                content = delta.content
                full_content += content
                yield content

        if chunk.choices and len(chunk.choices) > 0:
            full_response = chunk

    if full_response:
        try:
            if hasattr(full_response, 'usage') and full_response.usage:
                record_llm_usage(email, full_response)
        except Exception:
            pass
