"""
LLM helper — initializes LangChain chat models from user settings.
"""

import os
from typing import Optional

from llm_config import get_llm_config


def get_llm_for_user(user_email: Optional[str] = None, **kwargs):
    config = get_llm_config(user_email)
    provider = config["provider"]
    model = config["model"]
    api_key = config["api_key"]

    if not api_key:
        raise ValueError(
            "OpenAI API key is not configured. Set OPENAI_API_KEY in akkio-fastapi/.env and restart the server."
        )

    # Keep process env aligned so LangChain/OpenAI SDK never pick up a stale shell key.
    env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}.get(provider)
    if env_name:
        os.environ[env_name] = api_key

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, openai_api_key=api_key, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, anthropic_api_key=api_key, **kwargs)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **kwargs)

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, openai_api_key=api_key, **kwargs)


def get_llm_with_provider(provider: str, api_key: str, model: str, **kwargs):
    env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}.get(provider)
    if env_name:
        os.environ[env_name] = api_key
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, openai_api_key=api_key, **kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, anthropic_api_key=api_key, **kwargs)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **kwargs)
    raise ValueError(f"Unsupported provider: {provider}")
