"""Document upload + grounded chat MVP for Agentic Builder."""
from __future__ import annotations

from typing import Dict

from app_builder.services.fullstack_frontend_generator import (
    _page_header_tsx,
    _shell_tsx,
    build_theme_ts,
)


DOC_NAV = [
    ("/upload", "Upload documents"),
    ("/chat", "Document chat"),
]


def is_doc_chat_domain(requirement: str, prd: str = "", uiux: str = "") -> bool:
    text = "\n".join([requirement or "", prd or "", uiux or ""]).lower()
    upload = any(k in text for k in ("upload", "upload a file", "file upload", "upload documents"))
    chat = any(k in text for k in ("chat", "ask question", "questions", "q&a", "question", "chat with"))
    two_screens = "two screen" in text or "2 screen" in text
    if upload and chat:
        return True
    if two_screens and (upload or "document" in text or "doc" in text):
        return True
    if "document chat" in text or "chat with that" in text or "chat with the doc" in text:
        return True
    if "answer from doc" in text or "from doc properly" in text or "from the doc" in text:
        return True
    return False


def doc_chat_frontend_files(title: str, colors: Dict[str, str]) -> Dict[str, str]:
    safe_title = title.replace("\\", "\\\\").replace("'", "\\'")
    shell = _shell_tsx(title, DOC_NAV).replace(
        "Incoming Material",
        "Document intelligence",
    ).replace(
        "Automotive supplier quality · Incoming inspection &amp; release · IATF-aligned workflow",
        "Upload files · Ask questions · Answers grounded in your documents only",
    )
    return {
        "frontend/src/theme.ts": build_theme_ts(colors),
        "frontend/src/App.tsx": _doc_app_tsx(),
        "frontend/src/layout/AppShell.tsx": shell,
        "frontend/src/pages/UploadPage.tsx": _upload_page_tsx(),
        "frontend/src/pages/ChatPage.tsx": _chat_page_tsx(),
        "frontend/src/components/PageHeader.tsx": _page_header_tsx(),
        "frontend/src/components/DocumentList.tsx": _document_list_tsx(),
        "frontend/src/components/DocChatPanel.tsx": _doc_chat_panel_tsx(),
        "frontend/src/api/mock.ts": _doc_mock_ts(),
    }


def doc_chat_backend_files(title: str) -> Dict[str, str]:
    safe = (title or "Document Chat").replace('"', "'")[:80]
    return {
        "backend/models.py": _models_py(),
        "backend/schemas.py": _schemas_py(),
        "backend/document_service.py": _document_service_py(),
        "backend/routes.py": _routes_py(),
        "backend/seed.py": _seed_py(),
        "backend/main.py": _main_py(safe),
        "README.md": _readme(safe),
    }


def _doc_app_tsx() -> str:
    return """import { Navigate, Route, Routes } from 'react-router-dom';
import AppShell from './layout/AppShell';
import UploadPage from './pages/UploadPage';
import ChatPage from './pages/ChatPage';

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Navigate to="/upload" replace />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/chat" element={<ChatPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/upload" replace />} />
    </Routes>
  );
}
"""


def _upload_page_tsx() -> str:
    return r"""import { useCallback, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert, Box, Button, Card, CardContent, LinearProgress, Stack, Typography,
} from '@mui/material';
import CloudUploadOutlinedIcon from '@mui/icons-material/CloudUploadOutlined';
import PageHeader from '../components/PageHeader';
import DocumentList from '../components/DocumentList';
import { apiFetch, getApiBase } from '../api/client';

type Doc = { id: number; filename: string; size: number; status: string; uploaded_at?: string };

export default function UploadPage() {
  const qc = useQueryClient();
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState('');
  const { data, isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: () => apiFetch<{ items: Doc[] }>('/api/documents'),
  });
  const upload = useMutation({
    mutationFn: async (file: File) => {
      const base = getApiBase();
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${base}/api/documents/upload`, {
        method: 'POST',
        body: form,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Upload failed (${res.status})`);
      }
      return res.json();
    },
    onSuccess: () => {
      setError('');
      qc.invalidateQueries({ queryKey: ['documents'] });
    },
    onError: (e: Error) => setError(e.message),
  });

  const onFiles = useCallback((files: FileList | null) => {
    if (!files?.length) return;
    Array.from(files).forEach((f) => upload.mutate(f));
  }, [upload]);

  return (
    <Box>
      <PageHeader
        title="Upload documents"
        subtitle="Add PDF, Word, text, or markdown files. Content is indexed for grounded Q&A — no hallucinated answers."
      />
      <Card
        sx={{
          mb: 3,
          borderStyle: 'dashed',
          borderWidth: 2,
          borderColor: drag ? 'primary.main' : 'divider',
          bgcolor: drag ? 'action.hover' : 'background.paper',
          transition: 'border-color 0.2s, background 0.2s',
        }}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); onFiles(e.dataTransfer.files); }}
      >
        <CardContent sx={{ py: 5, textAlign: 'center' }}>
          <CloudUploadOutlinedIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
          <Typography variant="h6" sx={{ mb: 1 }}>Drop files here</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Supported: .txt, .md, .csv, .json, .pdf (text extraction when available)
          </Typography>
          <Button variant="contained" component="label" disabled={upload.isPending}>
            Choose files
            <input hidden type="file" multiple accept=".txt,.md,.csv,.json,.pdf,.doc,.docx" onChange={(e) => onFiles(e.target.files)} />
          </Button>
          {upload.isPending && <LinearProgress sx={{ mt: 3 }} />}
        </CardContent>
      </Card>
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      <Card>
        <CardContent>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
            <Typography variant="subtitle2">Your library</Typography>
            <Typography variant="caption" color="text.secondary">{data?.items?.length || 0} file(s)</Typography>
          </Stack>
          {isLoading ? <Typography>Loading…</Typography> : <DocumentList documents={data?.items || []} />}
        </CardContent>
      </Card>
    </Box>
  );
}
"""


def _chat_page_tsx() -> str:
    return r"""import { Box } from '@mui/material';
import PageHeader from '../components/PageHeader';
import DocChatPanel from '../components/DocChatPanel';

export default function ChatPage() {
  return (
    <Box>
      <PageHeader
        title="Document chat"
        subtitle="Ask questions about uploaded files. Answers cite only text found in your documents."
      />
      <DocChatPanel />
    </Box>
  );
}
"""


def _document_list_tsx() -> str:
    return r"""import { Chip, Stack, Typography } from '@mui/material';
import InsertDriveFileOutlinedIcon from '@mui/icons-material/InsertDriveFileOutlined';

type Doc = { id: number; filename: string; size: number; status: string; uploaded_at?: string };

export default function DocumentList({ documents }: { documents: Doc[] }) {
  if (!documents.length) {
    return <Typography color="text.secondary" variant="body2">No documents yet. Upload files to enable chat.</Typography>;
  }
  return (
    <Stack spacing={1}>
      {documents.map((d) => (
        <Stack key={d.id} direction="row" alignItems="center" spacing={1.5} sx={{ py: 1, borderBottom: 1, borderColor: 'divider' }}>
          <InsertDriveFileOutlinedIcon color="primary" fontSize="small" />
          <Typography variant="body2" sx={{ flex: 1, fontWeight: 600 }}>{d.filename}</Typography>
          <Chip size="small" label={d.status} color={d.status === 'INDEXED' ? 'success' : 'default'} />
          <Typography variant="caption" color="text.secondary">{Math.round((d.size || 0) / 1024)} KB</Typography>
        </Stack>
      ))}
    </Stack>
  );
}
"""


def _doc_chat_panel_tsx() -> str:
    return r"""import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Alert, Box, Button, Card, CardContent, Chip, Paper, Stack, TextField, Typography,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import { apiFetch } from '../api/client';

type AskResponse = {
  answer: string;
  sources: { document_id: number; filename: string; excerpt: string; score: number }[];
};

export default function DocChatPanel() {
  const [question, setQuestion] = useState('Summarize the main topics in my uploaded documents.');
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const { data: docs } = useQuery({ queryKey: ['documents'], queryFn: () => apiFetch<{ items: unknown[] }>('/api/documents') });
  const hasDocs = (docs?.items?.length || 0) > 0;

  const ask = async () => {
    setLoading(true);
    setErr('');
    try {
      const res = await apiFetch<AskResponse>('/api/chat/ask', {
        method: 'POST',
        body: JSON.stringify({ question }),
      });
      setAnswer(res);
    } catch (e: any) {
      setErr(e?.message || 'Could not get an answer');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardContent>
        {!hasDocs && (
          <Alert severity="warning" sx={{ mb: 2 }}>Upload at least one document before chatting.</Alert>
        )}
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
          <TextField
            fullWidth
            multiline
            minRows={2}
            label="Your question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <Button variant="contained" endIcon={<SendIcon />} onClick={ask} disabled={loading || !question.trim() || !hasDocs} sx={{ alignSelf: 'flex-start', px: 3 }}>
            {loading ? 'Thinking…' : 'Ask'}
          </Button>
        </Stack>
        {err && <Alert severity="error" sx={{ mt: 2 }}>{err}</Alert>}
        {answer && (
          <Paper variant="outlined" sx={{ mt: 3, p: 2, bgcolor: 'background.default' }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Answer</Typography>
            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>{answer.answer}</Typography>
            {!!answer.sources?.length && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>Sources (from your files only)</Typography>
                <Stack direction="row" flexWrap="wrap" gap={1}>
                  {answer.sources.map((s) => (
                    <Chip key={`${s.document_id}-${s.excerpt.slice(0, 12)}`} size="small" variant="outlined" label={`${s.filename} · ${Math.round(s.score * 100)}% match`} />
                  ))}
                </Stack>
              </Box>
            )}
          </Paper>
        )}
      </CardContent>
    </Card>
  );
}
"""


def _doc_mock_ts() -> str:
    return r"""type Json = Record<string, unknown>;
const docs: Json[] = [
  { id: 1, filename: 'supplier-quality-manual.txt', size: 48200, status: 'INDEXED', uploaded_at: new Date().toISOString() },
];
const chunks = [
  { document_id: 1, filename: 'supplier-quality-manual.txt', content: 'Incoming inspection requires diameter 280 ± 0.5 mm. Lots on hold must not be released without quality manager approval.', score: 0.92 },
];

export async function mockFetch<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {
  const method = (options.method || 'GET').toUpperCase();
  const clean = path.split('?')[0];
  if (clean.includes('/auth/login')) {
    return { access_token: 'mock-jwt', token_type: 'bearer', role: 'ADMIN' } as T;
  }
  if (method === 'GET' && clean.includes('/documents') && !clean.includes('upload')) {
    return { items: docs, total: docs.length } as T;
  }
  if (method === 'POST' && clean.includes('/documents/upload')) {
    const id = docs.length + 1;
    docs.push({ id, filename: `upload-${id}.txt`, size: 1200, status: 'INDEXED', uploaded_at: new Date().toISOString() });
    return { id, status: 'INDEXED' } as T;
  }
  if (method === 'POST' && (clean.includes('/chat/ask') || clean.includes('/ai/ask'))) {
    let question = '';
    try { question = JSON.parse(String(options.body || '{}')).question || ''; } catch { question = ''; }
    return {
      answer: `Grounded mock answer for: "${question}". From supplier-quality-manual.txt: diameter spec 280 ± 0.5 mm; hold lots require quality manager approval.`,
      sources: chunks,
    } as T;
  }
  return { ok: true, mocked: true } as T;
}
"""


def _main_py(title: str) -> str:
    return '''import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
from routes import router

app = FastAPI(title="%s", version="1.0.0", docs_url="/docs")

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
''' % title


def _models_py() -> str:
    return r'''from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, Index
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(64), nullable=False, default="VIEWER")
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    filename = Column(String(512), nullable=False)
    content_type = Column(String(128))
    size = Column(Integer, default=0)
    status = Column(String(32), default="INDEXED", index=True)
    full_text = Column(Text, default="")
    uploaded_by = Column(String(255))
    uploaded_at = Column(DateTime, default=datetime.utcnow, index=True)
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False, default=0)
    content = Column(Text, nullable=False)
    document = relationship("Document", back_populates="chunks")
    __table_args__ = (Index("ix_chunk_doc_idx", "document_id", "chunk_index"),)
'''


def _schemas_py() -> str:
    return r'''from pydantic import BaseModel, Field
from typing import List, Optional


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class LoginRequest(BaseModel):
    email: str
    password: str


class DocumentOut(BaseModel):
    id: int
    filename: str
    size: int
    status: str
    uploaded_at: Optional[str] = None

    class Config:
        from_attributes = True


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)


class SourceOut(BaseModel):
    document_id: int
    filename: str
    excerpt: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceOut]
'''


def _document_service_py() -> str:
    return r'''import os
import re
from typing import List, Tuple

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "data", "uploads"))


def ensure_upload_dir() -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return UPLOAD_DIR


def extract_text(filename: str, raw: bytes) -> str:
    lower = filename.lower()
    if lower.endswith((".txt", ".md", ".csv", ".json")):
        return raw.decode("utf-8", errors="replace")
    if lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(raw))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return raw.decode("utf-8", errors="replace")[:50000]
    return raw.decode("utf-8", errors="replace")[:50000]


def chunk_text(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if c]


def _tokenize(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]{3,}", (s or "").lower())}


def retrieve_chunks(question: str, rows: List[Tuple[int, str, str, str]], top_k: int = 5):
    """rows: (chunk_id, document_id, filename, content)"""
    q = _tokenize(question)
    if not q:
        return []
    scored = []
    for chunk_id, doc_id, filename, content in rows:
        words = _tokenize(content)
        if not words:
            continue
        overlap = len(q & words)
        score = overlap / max(len(q), 1)
        if score > 0:
            scored.append((score, chunk_id, doc_id, filename, content))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def grounded_answer(question: str, hits: List[tuple]) -> str:
    if not hits:
        return (
            "I could not find relevant text in your uploaded documents for that question. "
            "Upload more content or rephrase your question."
        )
    parts = []
    for score, _cid, _did, filename, content in hits[:3]:
        excerpt = content[:480].strip()
        parts.append(f"• From {filename}: \"{excerpt}\"")
    intro = "Based only on your uploaded documents:\n\n"
    return intro + "\n\n".join(parts)
'''


def _routes_py() -> str:
    return r'''import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from auth import create_access_token, get_current_user, verify_password
from database import get_db
from document_service import (
    chunk_text,
    ensure_upload_dir,
    extract_text,
    grounded_answer,
    retrieve_chunks,
)
from models import Document, DocumentChunk, User
from schemas import AskRequest, AskResponse, DocumentOut, LoginRequest, SourceOut, TokenResponse

router = APIRouter()


@router.post("/api/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.email, user.role)
    return TokenResponse(access_token=token, role=user.role)


@router.get("/api/documents")
def list_documents(db: Session = Depends(get_db)):
    rows = db.query(Document).order_by(Document.uploaded_at.desc()).all()
    items = [
        DocumentOut(
            id=d.id,
            filename=d.filename,
            size=d.size or 0,
            status=d.status,
            uploaded_at=d.uploaded_at.isoformat() if d.uploaded_at else None,
        )
        for d in rows
    ]
    return {"items": items, "total": len(items)}


@router.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")
    filename = file.filename or "upload.txt"
    text = extract_text(filename, raw)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    upload_dir = ensure_upload_dir()
    safe_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename.replace('/', '_')}"
    with open(os.path.join(upload_dir, safe_name), "wb") as fh:
        fh.write(raw)

    doc = Document(
        filename=filename,
        content_type=file.content_type,
        size=len(raw),
        status="INDEXED",
        full_text=text,
        uploaded_by=user.email if user else None,
    )
    db.add(doc)
    db.flush()
    for i, piece in enumerate(chunk_text(text)):
        db.add(DocumentChunk(document_id=doc.id, chunk_index=i, content=piece))
    db.commit()
    return {"id": doc.id, "filename": doc.filename, "status": doc.status, "chunks": len(chunk_text(text))}


def _ask_impl(payload: AskRequest, db: Session) -> AskResponse:
    rows = (
        db.query(DocumentChunk.id, DocumentChunk.document_id, Document.filename, DocumentChunk.content)
        .join(Document, Document.id == DocumentChunk.document_id)
        .all()
    )
    flat = [(r[0], r[1], r[2], r[3]) for r in rows]
    hits = retrieve_chunks(payload.question, flat, top_k=5)
    answer = grounded_answer(payload.question, hits)
    sources = [
        SourceOut(
            document_id=doc_id,
            filename=filename,
            excerpt=content[:220],
            score=round(float(score), 3),
        )
        for score, _cid, doc_id, filename, content in hits
    ]
    return AskResponse(answer=answer, sources=sources)


@router.post("/api/chat/ask", response_model=AskResponse)
def chat_ask(payload: AskRequest, db: Session = Depends(get_db)):
    return _ask_impl(payload, db)


@router.post("/api/ai/ask", response_model=AskResponse)
def ai_ask(payload: AskRequest, db: Session = Depends(get_db)):
    return _ask_impl(payload, db)
'''


def _seed_py() -> str:
    return r'''from database import Base, SessionLocal, engine
from models import User
from auth import hash_password

Base.metadata.create_all(bind=engine)


def run():
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            db.add(User(email="admin@example.com", hashed_password=hash_password("admin123"), role="ADMIN"))
            db.commit()
            print("Seeded admin user")
    finally:
        db.close()


if __name__ == "__main__":
    run()
'''


def _readme(title: str) -> str:
    return f"""# {title}

Document upload + grounded chat application.

## Screens

1. **Upload documents** — drag & drop or choose files; text is extracted and chunked.
2. **Document chat** — ask questions; answers cite only uploaded file content.

## Run

```bash
docker compose up --build
```

Demo login: `admin@example.com` / `admin123`

## API

- `POST /api/documents/upload` — multipart file upload
- `GET /api/documents` — list indexed files
- `POST /api/chat/ask` — `{{ "question": "..." }}` grounded response
"""
