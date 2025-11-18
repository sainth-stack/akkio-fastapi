import os
import re
from typing import Tuple
from openai import OpenAI


def get_openai_client():
    """Get OpenAI client with lazy initialization"""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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








