"""
LLM Helper Utility
Provides a unified interface to initialize LLMs from multiple providers
Automatically handles provider-specific initialization based on user settings
"""

from typing import Optional, Any
from llm_config import get_llm_config

def get_llm_for_user(user_email: Optional[str] = None, **kwargs):
    """
    Get initialized LLM instance for a user based on their settings.
    
    This function automatically:
    - Fetches user's provider preference (OpenAI, Anthropic, or Google)
    - Gets their API key (custom or system default)
    - Gets their preferred model
    - Initializes the appropriate LLM client
    
    Args:
        user_email: User email to look up settings. If None, uses system defaults.
        **kwargs: Additional parameters to pass to the LLM (temperature, max_tokens, etc.)
    
    Returns:
        Initialized LLM instance from the appropriate provider
        
    Example:
        # Basic usage
        llm = get_llm_for_user("user@example.com")
        response = llm.invoke("Hello, how are you?")
        
        # With additional parameters
        llm = get_llm_for_user("user@example.com", temperature=0.7, max_tokens=1000)
        response = llm.invoke("Write a story")
    """
    # Get user configuration
    config = get_llm_config(user_email)
    
    provider = config["provider"]
    model = config["model"]
    api_key = config["api_key"]
    
    # Initialize LLM based on provider
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI provider. "
                "Install it with: pip install langchain-openai"
            )
    
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                anthropic_api_key=api_key,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Anthropic provider. "
                "Install it with: pip install langchain-anthropic"
            )
    
    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Google provider. "
                "Install it with: pip install langchain-google-genai"
            )
    
    else:
        # Fallback to OpenAI for unknown providers
        print(f"Warning: Unknown provider '{provider}', falling back to OpenAI")
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "langchain-openai is required. "
                "Install it with: pip install langchain-openai"
            )


def get_llm_with_provider(provider: str, api_key: str, model: str, **kwargs):
    """
    Get initialized LLM instance for a specific provider.
    
    Use this function when you want to explicitly specify the provider
    rather than looking up user settings.
    
    Args:
        provider: Provider name ('openai', 'anthropic', or 'google')
        api_key: API key for the provider
        model: Model name to use
        **kwargs: Additional parameters for the LLM
    
    Returns:
        Initialized LLM instance
        
    Example:
        llm = get_llm_with_provider(
            provider="anthropic",
            api_key="sk-ant-...",
            model="claude-3-5-sonnet-20241022",
            temperature=0.7
        )
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, openai_api_key=api_key, **kwargs)
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, anthropic_api_key=api_key, **kwargs)
    
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **kwargs)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def invoke_llm_for_user(user_email: str, prompt: str, **llm_kwargs) -> str:
    """
    Convenience function to invoke an LLM with a simple prompt.
    
    Args:
        user_email: User email to look up settings
        prompt: The prompt to send to the LLM
        **llm_kwargs: Additional parameters for the LLM
    
    Returns:
        String response from the LLM
        
    Example:
        response = invoke_llm_for_user(
            "user@example.com",
            "Explain quantum computing in simple terms",
            temperature=0.7
        )
        print(response)
    """
    llm = get_llm_for_user(user_email, **llm_kwargs)
    response = llm.invoke(prompt)
    
    # Handle different response types
    if hasattr(response, 'content'):
        return response.content
    return str(response)


# Example usage functions for common patterns

def chat_completion_for_user(user_email: str, messages: list, **llm_kwargs) -> str:
    """
    Send a chat completion request with message history.
    
    Args:
        user_email: User email
        messages: List of message dicts with 'role' and 'content'
        **llm_kwargs: Additional LLM parameters
    
    Returns:
        String response
        
    Example:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Tell me a joke"}
        ]
        response = chat_completion_for_user("user@example.com", messages)
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    
    llm = get_llm_for_user(user_email, **llm_kwargs)
    
    # Convert message dicts to LangChain message objects
    langchain_messages = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'system':
            langchain_messages.append(SystemMessage(content=content))
        elif role == 'assistant' or role == 'ai':
            langchain_messages.append(AIMessage(content=content))
        else:  # user or human
            langchain_messages.append(HumanMessage(content=content))
    
    response = llm.invoke(langchain_messages)
    
    if hasattr(response, 'content'):
        return response.content
    return str(response)


def stream_llm_for_user(user_email: str, prompt: str, **llm_kwargs):
    """
    Stream LLM responses for real-time display.
    
    Args:
        user_email: User email
        prompt: The prompt to send
        **llm_kwargs: Additional LLM parameters
    
    Yields:
        String chunks as they arrive
        
    Example:
        for chunk in stream_llm_for_user("user@example.com", "Write a story"):
            print(chunk, end='', flush=True)
    """
    llm = get_llm_for_user(user_email, **llm_kwargs)
    
    for chunk in llm.stream(prompt):
        if hasattr(chunk, 'content'):
            yield chunk.content
        else:
            yield str(chunk)
