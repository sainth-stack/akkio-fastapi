"""
LLM Content Generator Backend - Generic template for ideas, LinkedIn posts, travel suggestions, etc.
Uses OpenAI API. Loads OPENAI_API_KEY from .env.
"""
import os
from pathlib import Path

def _load_env():
    try:
        from dotenv import load_dotenv
        base = Path(__file__).resolve().parent
        for _ in range(6):
            env_file = base / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                break
            if base.parent == base:
                break
            base = base.parent
        load_dotenv()
    except ImportError:
        pass

_load_env()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="LLM Content Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# gen_type: ideas | linkedin_post | travel | translate | general
GEN_PROMPTS = {
    "ideas": "You are a creative ideas generator. Given a topic, produce exactly {count} concise, actionable ideas. Return each idea on a new line, numbered 1. 2. 3. etc. No extra text.",
    "linkedin_post": "You are a LinkedIn content expert. Given a topic, write a professional, engaging LinkedIn post. Use a strong hook, short paragraphs (2-3 lines), end with a question or CTA. Include 3-5 relevant hashtags at the end.",
    "travel": "You are a travel expert. Given a destination or travel interest, produce exactly {count} travel suggestions, tips, or itinerary ideas. Return each on a new line, numbered.",
    "translate": "You are a translator. Translate the given text. Return only the translation, no explanations.",
    "general": "You are a helpful assistant. Given a topic or prompt, produce {count} relevant, concise responses. Return each on a new line, numbered.",
}


class GenerateRequest(BaseModel):
    topic: str
    count: Optional[int] = 5
    gen_type: Optional[str] = "ideas"


class GenerateResponse(BaseModel):
    topic: str
    gen_type: str
    content: list[str]
    raw_text: Optional[str] = None


def _generate_with_openai(topic: str, count: int, gen_type: str) -> tuple[list[str], str | None]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not set. Add it to .env in the project root or backend directory.",
        )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt_template = GEN_PROMPTS.get(gen_type, GEN_PROMPTS["ideas"])
        system_content = prompt_template.format(count=count)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": topic},
            ],
            max_tokens=600 if gen_type == "linkedin_post" else 500,
            temperature=0.8 if gen_type == "ideas" else 0.7,
        )
        text = (response.choices[0].message.content or "").strip()

        if gen_type == "linkedin_post":
            lines = text.split("\n")
            hashtag_line = None
            for i, line in enumerate(lines):
                if line.strip().startswith("#"):
                    hashtag_line = line.strip()
                    break
            post_lines = [l for l in lines if not l.strip().startswith("#")] if hashtag_line else lines
            post = "\n".join(post_lines).strip()
            result = [post] if post else [text]
            return result, text
        elif gen_type == "translate":
            return [text] if text else ["No translation."], text
        else:
            ideas = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line[0].isdigit() and (". " in line or ") " in line):
                    line = line.split(". ", 1)[-1] if ". " in line else line.split(") ", 1)[-1]
                if line:
                    ideas.append(line)
            return ideas[:count] if ideas else [text] if text else ["No content generated."], text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "llm-content-generator"}


# Keeps /generate-ideas for backward compatibility
@app.post("/generate-ideas", response_model=GenerateResponse)
def generate_ideas(req: GenerateRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    count = max(1, min(10, req.count or 5))
    raw_type = (req.gen_type or "ideas").lower().strip()
    gen_type = raw_type.replace("-", "_").replace(" ", "_")
    if gen_type not in GEN_PROMPTS:
        gen_type = "ideas"
    content, raw = _generate_with_openai(req.topic.strip(), count, gen_type)
    return GenerateResponse(topic=req.topic, gen_type=gen_type, content=content, raw_text=raw)


# Generic endpoint
@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    return generate_ideas(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)
