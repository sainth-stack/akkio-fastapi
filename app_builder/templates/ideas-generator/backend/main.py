"""
Ideas Generator Backend - Uses OpenAI API to generate ideas from a topic.
Loads OPENAI_API_KEY from .env (searches current dir, project root, and parent).
"""
import os
from pathlib import Path

# Load .env - searches backend/, project root, and parents up to 5 levels (finds akkio-fastapi/.env)
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
        load_dotenv()  # cwd override
    except ImportError:
        pass

_load_env()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Ideas Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class IdeasRequest(BaseModel):
    topic: str
    count: Optional[int] = 5


class IdeasResponse(BaseModel):
    topic: str
    ideas: list[str]


def _generate_ideas_with_openai(topic: str, count: int) -> list[str]:
    """Generate ideas using OpenAI API. Requires OPENAI_API_KEY in .env"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not set. Add it to .env in the project root or backend directory.",
        )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a creative ideas generator. Given a topic, produce exactly {count} concise, actionable ideas. Return each idea on a new line, numbered 1. 2. 3. etc. No extra text.",
                },
                {"role": "user", "content": topic},
            ],
            max_tokens=500,
            temperature=0.8,
        )
        text = (response.choices[0].message.content or "").strip()
        ideas = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and (". " in line or ") " in line):
                line = line.split(". ", 1)[-1] if ". " in line else line.split(") ", 1)[-1]
            if line:
                ideas.append(line)
        return ideas[:count] if ideas else [text] if text else ["No ideas generated."]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "ideas-generator"}


@app.post("/generate-ideas", response_model=IdeasResponse)
def generate_ideas(req: IdeasRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    count = max(1, min(10, req.count or 5))
    ideas = _generate_ideas_with_openai(req.topic.strip(), count)
    return IdeasResponse(topic=req.topic, ideas=ideas)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5004, reload=True)
