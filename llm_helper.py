"""
LLM helper — initializes LangChain chat models from user settings.
"""

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
