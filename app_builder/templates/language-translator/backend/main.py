"""
Language Translator Backend - FastAPI API for text translation.
Uses googletrans (or similar) for translation. Fallback: placeholder logic.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Language Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str
    source_lang: Optional[str] = "auto"
    target_lang: str = "en"


class TranslateResponse(BaseModel):
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str


# Supported languages mapping
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh-cn": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
}


def _translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Perform translation. Uses googletrans if available, else placeholder."""
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, src=source_lang if source_lang != "auto" else None, dest=target_lang)
        return result.text
    except ImportError:
        # Fallback when googletrans not installed
        return f"[Translated: {text[:50]}...] (Install googletrans: pip install googletrans==4.0.0-rc1)"


@app.get("/health")
def health():
    return {"status": "ok", "service": "language-translator"}


@app.get("/languages")
def list_languages():
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if req.target_lang not in SUPPORTED_LANGUAGES and req.target_lang != "auto":
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {req.target_lang}")
    try:
        translated = _translate_text(req.text, req.source_lang or "auto", req.target_lang)
        return TranslateResponse(
            original_text=req.text,
            translated_text=translated,
            source_lang=req.source_lang or "auto",
            target_lang=req.target_lang,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5004, reload=True)
