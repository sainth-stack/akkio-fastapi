"""
LLM Configuration Utility
Provides a common function to get LLM configuration (API key, model, and provider)
Server API keys always come from akkio-fastapi/.env (not shell env or DB overrides).
User DB settings may override provider/model only.
Supports: OpenAI, Anthropic (Claude), and Google (Gemini)
"""
import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import dotenv_values, load_dotenv

from db import PostgresDatabase

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH, override=True)

_PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# Default LLM configuration (aligned with app defaults and Settings UI)
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"

# Provider-specific environment variables (read at runtime via get_default_api_key)

# Provider models mapping with all latest models (Updated January 2026)
PROVIDER_MODELS = {
    "openai": {
        "default": "gpt-4o-mini",
        "models": [
            # GPT-5 Series (Latest - Released August 2025+)
            "gpt-5.2",                      # Most capable (December 2025)
            "gpt-5.2-codex",                # Agentic coding specialization (January 2026)
            "gpt-5.1",                      # Instant & Thinking modes (November 2025)
            "gpt-5",                        # Flagship (August 2025)
            "gpt-5-mini",                   # Lighter version
            "gpt-5-nano",                   # Smallest version
            "gpt-5-codex",                  # Optimized for coding
            
            # GPT-4.5 Series (Released February 2025 - Deprecated, superseded by GPT-5)
            "gpt-4.5",
            "gpt-4.5-mini",
            "gpt-4.5-nano",
            
            # GPT-4.1 Series (Released April 2025)
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            
            # GPT-4o Series (Being deprecated Feb 2026)
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13",
            "gpt-4o-mini-2024-07-18",
            "chatgpt-4o-latest",            # Retiring Feb 16, 2026
            
            # GPT-4 Turbo Series
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            
            # GPT-4 Series
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-0314",
            
            # GPT-3.5 Turbo Series
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-instruct",
            
            # O1 Series (Reasoning models)
            "o1-preview",
            "o1-mini",
            "o1-preview-2024-09-12",
            "o1-mini-2024-09-12",
            
            # Specialized Models (Realtime & Audio)
            "gpt-realtime-mini",
            "gpt-audio-mini",
            "gpt-4o-mini-tts",
            "gpt-4o-realtime-preview",
            "gpt-4o-audio-preview",
        ]
    },
    "anthropic": {
        "default": "claude-opus-4-5",
        "models": [
            # Claude 4.5 Series (Latest - January 2026)
            "claude-opus-4-5",              # Most capable (200K context)
            "claude-sonnet-4-5",            # Balanced, great for agents
            "claude-haiku-4-5",             # Fast and efficient
            
            # Claude 4.1 Series
            "claude-opus-4-1",
            "claude-sonnet-4-1",
            "claude-haiku-4-1",
            
            # Claude 4 Series
            "claude-opus-4",
            "claude-sonnet-4",
            "claude-haiku-4",
            
            # Claude 3.5 Series
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            
            # Claude 3 Series (Opus 3 retired January 2026)
            "claude-3-opus-latest",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            
            # Claude 2 Series (Legacy)
            "claude-2.1",
            "claude-2.0",
            
            # Claude Instant (Legacy)
            "claude-instant-1.2",
        ]
    },
    "google": {
        "default": "gemini-3-flash",
        "models": [
            # Gemini 3 Series (Latest - November/December 2025)
            "gemini-3-pro",                 # Most capable (Nov 2025)
            "gemini-3-flash",               # Fast and efficient (Dec 2025)
            "gemini-3-deepthink",           # Advanced reasoning (Nov 2025)
            "gemini-3-pro-image",           # Nano Banana Pro - Image generation
            
            # Gemini 2.5 Series (Mid-2025 - Still active)
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-image",
            
            # Gemini 2.0 Series (Being deprecated Feb 2026)
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-thinking-exp",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            
            # Gemini 1.5 Pro Series
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-002",
            "gemini-1.5-pro-001",
            "gemini-1.5-pro-exp-0827",
            "gemini-1.5-pro-exp-0801",
            
            # Gemini 1.5 Flash Series
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash-8b",
            "gemini-1.5-flash-8b-latest",
            "gemini-1.5-flash-8b-001",
            "gemini-1.5-flash-8b-exp-0827",
            "gemini-1.5-flash-exp-0827",
            
            # Gemini 1.0 Pro Series
            "gemini-1.0-pro",
            "gemini-1.0-pro-latest",
            "gemini-1.0-pro-001",
            "gemini-1.0-pro-vision",
            
            # Experimental
            "gemini-exp-1206",
            "gemini-exp-1121",
        ]
    }
}

def _read_env_file() -> dict:
    """Read akkio-fastapi/.env directly so shell OPENAI_API_KEY cannot override it."""
    if not _ENV_PATH.is_file():
        return {}
    values = dotenv_values(_ENV_PATH) or {}
    return {k: v for k, v in values.items() if v is not None}


def get_default_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider from akkio-fastapi/.env first, then process env."""
    env_name = _PROVIDER_ENV_KEYS.get(provider)
    if not env_name:
        return None
    file_env = _read_env_file()
    key = (file_env.get(env_name) or "").strip()
    if key:
        return key
    return (os.getenv(env_name) or "").strip() or None


def get_llm_config(user_email: Optional[str] = None) -> Dict[str, str]:
    """
    Get LLM configuration for a user.

    API keys always come from ``akkio-fastapi/.env`` (never from ``llm_settings`` or shell env).

    Model/provider: user ``llm_settings`` override defaults when present.

    Args:
        user_email: User email to look up custom settings

    Returns:
        Dictionary with 'provider', 'api_key', and 'model' keys
    """
    config = {
        "provider": DEFAULT_PROVIDER,
        "api_key": get_default_api_key(DEFAULT_PROVIDER),
        "model": DEFAULT_MODEL
    }

    if not user_email:
        return config

    try:
        db = PostgresDatabase()
        db.ensure_connection()

        user_settings = db.get_llm_settings(user_email)

        if user_settings:
            provider = user_settings.get("provider") or "openai"
            config["provider"] = provider
            config["api_key"] = get_default_api_key(provider)

            if user_settings.get("model_name"):
                config["model"] = user_settings["model_name"]
            else:
                config["model"] = PROVIDER_MODELS.get(provider, {}).get("default", DEFAULT_MODEL)

        db.close()
    except Exception as e:
        print(f"Error fetching LLM config for user {user_email}: {e}")

    return config


def get_provider(user_email: Optional[str] = None) -> str:
    """
    Convenience function to get just the provider.
    
    Args:
        user_email: User email to look up custom settings
        
    Returns:
        Provider string (openai, anthropic, or google)
    """
    return get_llm_config(user_email)["provider"]


def get_api_key(user_email: Optional[str] = None) -> str:
    """
    Convenience function to get just the API key.
    
    Args:
        user_email: User email to look up custom settings
        
    Returns:
        API key string
    """
    return get_llm_config(user_email)["api_key"]


def get_model_name(user_email: Optional[str] = None) -> str:
    """
    Convenience function to get just the model name.
    
    Args:
        user_email: User email to look up custom settings
        
    Returns:
        Model name string
    """
    return get_llm_config(user_email)["model"]


def get_provider_models(provider: str = None) -> list:
    """
    Get available models for a provider.
    
    Args:
        provider: Provider name (openai, anthropic, google). If None, returns all.
        
    Returns:
        List of model names or dict of all providers' models
    """
    if provider:
        return PROVIDER_MODELS.get(provider, {}).get("models", [])
    return PROVIDER_MODELS


def get_default_model_for_provider(provider: str) -> str:
    """
    Get the default model for a specific provider.
    
    Args:
        provider: Provider name (openai, anthropic, google)
        
    Returns:
        Default model name for the provider
    """
    return PROVIDER_MODELS.get(provider, {}).get("default", DEFAULT_MODEL)
